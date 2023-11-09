#include "mfem.hpp"
#include "object.h"
#include "object_cc.hh"
#include "ddcMalloc.h"
#include "pio.h"
#include "pioFixedRecordHelper.h"
#include "units.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <cassert>
#include <memory>
#include <set>
#include <ctime>
#include <chrono>
#include <cstdio>
#include <dirent.h>
#include <regex.h>
#include <unistd.h>
#include <sys/stat.h>
#include "util.hpp"
#include "MatrixElementPiecewiseCoefficient.hpp"
#include "cardiac_coefficients.hpp"


//#define StartTimer(x)
//#define EndTimer()

using namespace mfem;

MPI_Comm COMM_LOCAL = MPI_COMM_WORLD;

class Timeline
{
 public:
   Timeline(double dt, double duration)
   {
      maxTimesteps_ = round(duration/dt);
      dt_ = duration/maxTimesteps_;
   }
   int maxTimesteps() const { return maxTimesteps_; };
   double dt() const { return dt_; }
   double maxTime() const { return dt_*maxTimesteps_; }
   double realTimeFromTimestep(int timestep) const
   {
      return timestep*dt_;
   }
   int timestepFromRealTime(double realTime) const
   {
      return round(realTime/dt_);
   }
 private:
   double dt_;
   int maxTimesteps_;
};


int main(int argc, char *argv[])
{  
   MPI_Init(NULL,NULL);   // without specific command line
   int num_ranks, my_rank;
   MPI_Comm_size(MPI_COMM_WORLD, &num_ranks); // number of processes in COMM_LOCAL
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); 

   units_internal(1e-3, 1e-9, 1e-3, 1e-3, 1, 1e-9, 1);
   units_external(1e-3, 1e-9, 1e-3, 1e-3, 1, 1e-9, 1);

   if (my_rank == 0)
   {
      std::cout << "Initializing with " << num_ranks << " MPI ranks." << std::endl;
   }
   
   int order = 1;

   std::vector<std::string> objectFilenames;
   if (argc == 1)
      //push_back function is used to add an element to the end of the vector.
      objectFilenames.push_back("femheart.data"); 

   for (int iargCursor=1; iargCursor<argc; iargCursor++)
      objectFilenames.push_back(argv[iargCursor]);

   if (my_rank == 0) {
      for (int ii=0; ii<objectFilenames.size(); ii++)
	 object_compilefile(objectFilenames[ii].c_str());
   }
   object_Bcast(0,MPI_COMM_WORLD);

   OBJECT* obj = object_find("femheart", "HEART");
   assert(obj != NULL);


   // Read shared global mesh
   mfem::Mesh *mesh = ecg_readMeshptr(obj, "mesh");


   int dim = mesh->Dimension();

   //Fill in the MatrixElementPiecewiseCoefficients
   std::vector<double> sigma_m;
   objectGet(obj,"sigma_m",sigma_m);

   double dt;
   objectGet(obj,"dt",dt,"0.01 ms");
   double Bm;
   objectGet(obj,"Bm",Bm,"140"); // 1/mm
   double Cm;
   objectGet(obj,"Cm",Cm,"0.01"); // 1 uF/cm^2 = 0.01 uF/mm^2
 
   std::string reactionName;
   objectGet(obj, "reaction", reactionName, "BetterTT06");

   std::string outputDir;
   objectGet(obj, "outdir", outputDir, ".");
   
   double endTime;
   objectGet(obj, "end_time", endTime, "0 ms");

   double outputRate;
   objectGet(obj, "output_rate", outputRate, "1 ms");

   //double checkpointRate;
   //objectGet(obj, "checkpoint_rate", checkpointRate, "100 ms");

   double initVm;
   objectGet(obj, "init_vm", initVm, "-83");


   StimulusCollection stims(dt);
   {
      std::vector<std::string> stimulusNames;
      objectGet(obj, "stimulus", stimulusNames);
      for (auto name : stimulusNames)
      {
         OBJECT* stimobj = object_find(name.c_str(), "STIMULUS");
         assert(stimobj != NULL);
         int numTimes;
         objectGet(stimobj, "n", numTimes, "1");
         double bcl;
         objectGet(stimobj, "bcl", bcl, "0 ms");
         assert(numTimes == 1 || bcl != 0);
         double startTime;
         objectGet(stimobj, "start", startTime, "0 ms");
         double duration;
         objectGet(stimobj, "duration", duration, "1 ms");
         double strength;
         objectGet(stimobj, "strength", strength, "0"); //uA/uF
         assert(strength >= 0);
         std::string location;
         objectGet(stimobj, "where", location, "");
         assert(!location.empty());
         OBJECT* locobj = object_find(location.c_str(), "REGION");
         assert(locobj != NULL);
         std::string regionType;
         objectGet(locobj, "type", regionType, "");
         assert(!regionType.empty());
         shared_ptr<StimulusLocation> stimLoc;
         if (regionType == "ball")
         {
            std::vector<double> center;
            objectGet(locobj, "center", center);
            assert(center.size() == 3);
            double radius;
            objectGet(locobj, "radius", radius, "-1");
            assert(radius >= 0);
            stimLoc = std::make_shared<CenterBallStimulus>(center[0],center[1],center[2],radius);
         }
         else if (regionType == "box")
         {
            std::vector<double> lower;
            objectGet(locobj, "lower", lower);
            assert(lower.size() == 3);
            vector<double> upper;
            objectGet(locobj, "upper", upper);
            assert(upper.size() == 3);
            stimLoc = std::make_shared<BoxStimulus>
               (lower[0], upper[0],
                lower[1], upper[1],
                lower[2], upper[2]);
         }
         shared_ptr<StimulusWaveform> stimWave(new SquareWaveform());
         stims.add(Stimulus(numTimes, startTime, duration, bcl, strength, stimLoc, stimWave));
      }
   }
   
   Timeline timeline(dt, endTime);   

   /*
     Ok, we're solving:

     div(sigma_m*grad(Vm)) = Bm*Cm*(dVm/dt + Iion - Istim)

     time in ms
     space in mm
     Vm in mV
     Iion in uA/uF
     Istim in uA/uF
     Cm in uF/mm^2
     Bm in 1/mm
     sigma in mS/mm

     To solve this, I use a BackEuler method:

     div(sigma_m*grad(Vm_new) = Bm*Cm*[(Vm_new-Vm_old)/dt + Iion - Istim

     with some algebra, that comes to

     {1+dt/(Bm*Cm)*(div sigma_m*grad)}*Vm_new = {1}*Vm_old - dt*Iion + dt*Istim

     Each {} is a matrix that is done with FEM.

     sigma_m = sigma_e_diagonal*sigma_i_diagonal/(sigma_e_diagonal+sigma_i_diagonal)

     This is the monodomain conductivity.  It really only approximates the bidomain conductivity if the ratio
     of sigma_e_tensor = k*sigma_i_tensor.  When this happens, you can remove Phi_e from the equations and
     end up with the equation listed above. 

   */

   
   mesh->SetAttributes();
   //If I read correctly, pmeshpart will now point to an integer array
   //containing a partition ID (rank!) for every element ID.
   int *pmeshpart = mesh->GeneratePartitioning(num_ranks);


   //Go through all the elements and label the partitioning for the vertices
   std::vector<set<int> > pvertset(mesh->GetNV());
   for (int ielem=0; ielem<mesh->GetNE(); ielem++)
   {
      Array<int> verts;
      mesh->GetElementVertices(ielem, verts);
      for (int ivert=0; ivert<verts.Size(); ivert++)
      {
         pvertset[verts[ivert]].insert(pmeshpart[ielem]);
      }
   }

   std::vector<int> local_extents(num_ranks+1);
   {
      std::vector<int> local_counts(num_ranks, 0);
      for(int i=0; i<mesh->GetNV(); i++)
      {
         if ( ! pvertset[i].empty())
         {
            local_counts[*(pvertset[i].begin())]++;
         }
      }
      local_extents[0] = 0;
      for (int irank=0; irank<num_ranks; irank++)
      {
         local_extents[irank+1] = local_extents[irank]+local_counts[irank];
      }
   }

   std::vector<int> globalvert_from_ranklookup(local_extents[num_ranks]);
   std::vector<int> ghostlocalvert_from_ranklookup(local_extents[num_ranks]);
   {
      std::vector<int> cursor_ghostlocal_from_rank(num_ranks, 0);
      std::vector<int> cursor_ranklookup_from_rank = local_extents;
      for(int i=0; i<mesh->GetNV(); i++)
      {
         if ( ! pvertset[i].empty())
         {
            int irank = *(pvertset[i].begin());
            int ranklookup = cursor_ranklookup_from_rank[irank]++;
            int globalvert = i;
            int ghostlocal = cursor_ghostlocal_from_rank[irank];
            globalvert_from_ranklookup[ranklookup] = globalvert;
            ghostlocalvert_from_ranklookup[ranklookup] = ghostlocal;
            for (const int used_by_this_rank : pvertset[i])
            {
               cursor_ghostlocal_from_rank[used_by_this_rank]++;
            }
         }
      }
   }

   //Get the element material types for each index.
   std::vector<int> material_from_ranklookup(local_extents[num_ranks]);
   {
      std::vector<int> element_from_globalvert(mesh->GetNV(), mesh->GetNE());
      for (int ielem=0; ielem<mesh->GetNE(); ielem++)
      {
         Array<int> verts;
         mesh->GetElementVertices(ielem, verts);
         for (int ivert=0; ivert<verts.Size(); ivert++)
         {
            element_from_globalvert[verts[ivert]] = std::min(element_from_globalvert[verts[ivert]], ielem);
         }
      }
      std::vector<int> cursor_ranklookup_from_rank = local_extents;
      for(int i=0; i<mesh->GetNV(); i++)
      {
         if ( ! pvertset[i].empty())
         {
            int irank = *(pvertset[i].begin());
            int ranklookup = cursor_ranklookup_from_rank[irank]++;
            int globalvert = i;

            int ielem = element_from_globalvert[globalvert];
            material_from_ranklookup[ranklookup] = mesh->GetElement(ielem)->GetAttribute();
         }
      }
   }
   
   if (my_rank == 0)
   {
      for(int i=0; i<num_ranks; i++) {
         std::cout << "Rank " << i << " has " << local_extents[i+1]-local_extents[i] << " nodes!" << std::endl;
      }
   }

   //Define a parallel mesh by a partitioning of the serial mesh.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh, pmeshpart);
   
   //Build a new FEC and corresponding FES
   if (my_rank == 0) { std::cout << "Creating FEC" << std::endl; }
   FiniteElementCollection *fec = new H1_FECollection(order, dim);
   ParFiniteElementSpace *pfespace = new ParFiniteElementSpace(pmesh, fec);
   std::cout << "[" << my_rank << "] Number of finite element unknowns: "
	     << pfespace->GetTrueVSize() << std::endl;

   //Determine the list of true (i.e. conforming) essential boundary DOFs
   //Essential true degrees of freedom, "true" takes into account shared vertices.
   Array<int> ess_tdof_list;   
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 0;
      pfespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   //Define the solution vector gf_Vm as a finite element grid function
   //corresponding to pfespace. Initialize gf_Vm with initial value initVm.
   ParGridFunction gf_Vm(pfespace);
   gf_Vm = initVm;

   //Read fiber quaternions from file
   std::shared_ptr<GridFunction> flat_fiber_quat;
   ecg_readGF(obj, "fibers", mesh, flat_fiber_quat);
   std::shared_ptr<ParGridFunction> fiber_quat;
   fiber_quat = std::make_shared<mfem::ParGridFunction>(pmesh, flat_fiber_quat.get(), pmeshpart);

   //Load conductivity data
   MatrixElementPiecewiseCoefficient sigma_m_coeffs(fiber_quat);
   { 
      Vector sigma_m_vec(3);
      for (int jj=0; jj<3; jj++)
      {
         double value = sigma_m[jj]*dt/2/Bm/Cm;
         sigma_m_vec[jj] = value;
      }
      sigma_m_coeffs.heartConductivities_[1] = sigma_m_vec;
   }
   //Set up the bilinear form a(.,.) on the finite element space
   //orresponding to the Laplacian operator -Delta, by adding the Diffusion
   //domain integrator.
   //NOTICE THE FLIP IN SIGNS FOR SIGMA!  This is on purpose, Diffusion does -div(sigma*grad)

   //Set up left-hand side matrix LHS_mat
   ConstantCoefficient one(1.0);
   ParBilinearForm *a = new ParBilinearForm(pfespace); 
   a->AddDomainIntegrator(new DiffusionIntegrator(sigma_m_coeffs));
   a->AddDomainIntegrator(new MassIntegrator(one)); 
   a->Assemble();
   HypreParMatrix LHS_mat; 
   a->FormSystemMatrix(ess_tdof_list,LHS_mat);

   //Set up right-hand side matrix RHS_mat
   ParBilinearForm *b = new ParBilinearForm(pfespace); 
   b->AddDomainIntegrator(new MassIntegrator(one));
   b->Update(pfespace);  
   b->Assemble();
   HypreParMatrix RHS_mat; 
   b->FormSystemMatrix(ess_tdof_list, RHS_mat);

   //Set up the Iion term
   ConstantCoefficient coe_dt(dt);
   ParBilinearForm *Iion = new ParBilinearForm(pfespace); 
   Iion->AddDomainIntegrator(new MassIntegrator(coe_dt));
   Iion->Update(pfespace);  
   Iion->Assemble();
   HypreParMatrix Iion_mat; 
   Iion->FormSystemMatrix(ess_tdof_list, Iion_mat);

   //Set up the stimulus term
   ParLinearForm *c = new ParLinearForm(pfespace);
   c->AddDomainIntegrator(new DomainLFIntegrator(stims));

   //Set up the solve    
   HyprePCG pcg(LHS_mat);
   pcg.SetTol(1e-12);
   pcg.SetMaxIter(2000);
   pcg.SetPrintLevel(2);
   HypreSolver *M_test = new HypreBoomerAMG(LHS_mat);
   pcg.SetPreconditioner(*M_test);

   //Prepare for reaction term
   ThreadServer& threadServer = ThreadServer::getInstance();
   ThreadTeam defaultGroup = threadServer.getThreadTeam(vector<unsigned>());

   std::vector<std::string> reactionNames;
   reactionNames.push_back(reactionName);

   std::vector<int> cellTypes;
   for (int ranklookup=local_extents[my_rank]; ranklookup<local_extents[my_rank+1]; ranklookup++)
   {
      cellTypes.push_back(material_from_ranklookup[ranklookup]);
   }

   ReactionWrapper reactionWrapper(dt,reactionNames,defaultGroup,cellTypes);
   reactionWrapper.Initialize();
   cellTypes.clear();
   reactionNames.clear();

   Vector v_new(pfespace->GetTrueVSize());
   Vector actual_b(pfespace->GetTrueVSize());
   Vector actual_old(pfespace->GetTrueVSize());
   Vector actual_Iion(pfespace->GetTrueVSize());

   bool first=true;

   v_new = reactionWrapper.getVmReadonly();

   // Initialize paraview 
   ParaViewDataCollection pd("V_m", pmesh);
   pd.SetPrefixPath("ParaView");
   pd.RegisterField("solution", &gf_Vm);
   pd.SetLevelsOfDetail(order);
   pd.SetDataFormat(VTKFormat::BINARY);
   pd.SetHighOrderOutput(true);
   pd.SetCycle(0);
   pd.SetTime(0.0);
   pd.Save();
   
   int itime=0;
   clock_t time_start = clock(); //record time start

   while (itime != timeline.maxTimesteps()) 
   {  
      
      if (my_rank == 0)
      {  
         double time = (double)(clock()-time_start)/CLOCKS_PER_SEC;
         std::cout << "times =" << time << "seconds." << std::endl;
         std::cout << "time = " << timeline.realTimeFromTimestep(itime) << std::endl;
      }

      //calculate the ionic contribution.
      v_new = reactionWrapper.getVmReadwrite() ; //should be a memcpy
      reactionWrapper.Calc();
  
      //Compute stimulus contribution
      stims.updateTime(timeline.realTimeFromTimestep(itime));
      c->Update();
      c->Assemble();

      //Form Lv=R (L=LHS_mat, v=v_new(gf_Vm), R=actual_b(c))
      a->FormLinearSystem(ess_tdof_list, gf_Vm, *c, LHS_mat, v_new, actual_b, 1);

      //compute the RHS matrix Iion contribution
      //actual_old = RHS_mat * v_new + Iion_mat * Iion  
      RHS_mat.Mult(v_new, actual_old);
      actual_b += actual_old;

      Iion_mat.Mult(reactionWrapper.getIionReadonly(),actual_old);
      actual_b += actual_old;

      //Solve the linear system LHS_mat * v_new = b
      pcg.Mult(actual_b, v_new);

      //Recover the solution v_new as a grid function gf_Vm
      a->RecoverFEMSolution(v_new, *c, gf_Vm);

      itime++;

      //output if appropriate
      if ((itime % timeline.timestepFromRealTime(outputRate)) == 0)
      {
         pd.SetCycle(itime);
         pd.SetTime(timeline.realTimeFromTimestep(itime));
         pd.Save();
      }

      first=false;
   }

   // 14. Free the used memory.
   delete M_test;
   delete a;
   delete b;
   delete c;
   delete pfespace;
   if (order > 0) { delete fec; }
   delete mesh, pmesh, pmeshpart;
   return 0;
}
