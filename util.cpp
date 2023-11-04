
#include "util.hpp"
#include "object_cc.hh"
#include "pio.h"
#include "pioFixedRecordHelper.h"
#include <fstream>
#include <cassert>



std::set<int> readSet(std::istream& stream) {
   std::set<int> retval;
   while(1) {
      int value;
      stream >> value;
      if (!stream) {break;}
      retval.insert(value);
   }
   return retval;
}

std::set<int> ecg_readSet(OBJECT* obj, const std::string dataname) {
   std::string filename;
   objectGet(obj,dataname,filename,"");

   std::ifstream infile(filename);
   std::set<int> retval = readSet(infile);
#ifdef DEBUG
   std::cout << "Read " << retval.size() << " elements from "
	     << filename << std::endl;
#endif
   return retval;
}

std::unordered_map<int,int> readMap(std::istream& stream) {
   std::unordered_map<int,int> retval;
   int i = 0;
   while(1) {
      int value;
      stream >> value;
      if (!stream) {break;}
      retval[i++] = value;
   }
   return retval;
}

std::unordered_map<int,int> ecg_readMap(OBJECT* obj, const std::string dataname) {
   std::string filename;
   objectGet(obj,dataname,filename,"");

   std::ifstream infile(filename);
   std::unordered_map<int,int> retval = readMap(infile);
#ifdef DEBUG
   std::cout << "Read " << retval.size() << " elements from "
	     << filename << std::endl;
#endif
   return retval;
}

void ecg_readGF(OBJECT* obj, const std::string dataname,
		mfem::Mesh* mesh, std::shared_ptr<mfem::GridFunction>& sp_gf) {
   std::string filename;
   objectGet(obj,dataname,filename,"");

   std::ifstream infile(filename);
   sp_gf = std::make_shared<mfem::GridFunction>(mesh, infile);
}

mfem::Mesh* ecg_readMeshptr(OBJECT* obj, const std::string dataname) {
   std::string filename;
   objectGet(obj, dataname, filename, "");

   // generate_edges=1, refine=1 (fix_orientation=true by default)
   return new mfem::Mesh(filename.c_str(), 1, 1);
}

// If rank==0, then object_bcast()

void ecg_process_args(int argc, char* argv[]) {
   std::vector<std::string> objectFilenames;
   if (argc == 1)
      objectFilenames.push_back("ecg.data");

   for (int iargCursor=1; iargCursor<argc; iargCursor++)
      objectFilenames.push_back(argv[iargCursor]);

   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   if (rank == 0) {
      for (int ii=0; ii<objectFilenames.size(); ii++)
	 object_compilefile(objectFilenames[ii].c_str());
   }
   object_Bcast(0,MPI_COMM_WORLD);
}

// Read sensors
std::unordered_map<int,int> ecg_readInverseMap(OBJECT* obj, const std::string dataname) {
   std::string filename;
   objectGet(obj,dataname,filename,"");
   std::ifstream sensorStream(filename);

   std::unordered_map<int,int> gfFromGid;
   //read in the sensor file
   {
      int token = 0;
      do {
	 //read in from the sensor file
	 int sensorGid;
	 sensorStream >> sensorGid;
	 if (!sensorStream) break;
	 gfFromGid[sensorGid] = token++;
      } while(1);
   }

   return gfFromGid;
}

// This is really awful.  Too specific to be general, but presented as if it could be.
std::map<int,std::string> ecg_readAssignments(OBJECT* obj, const std::string dataname) {
   std::string filename;
   objectGet(obj,dataname,filename,"");
   std::ifstream electrodeStream(filename);

   std::map<int,std::string> electrodeFromGid;
   do {
      int gid;
      std::string label;
      electrodeStream >> label; // First column becomes the electrode name
      electrodeStream >> gid;   // Second column gets discarded
      electrodeStream >> gid;   // Third column becomes the GID
      if (!electrodeStream) break;
      electrodeFromGid[gid] = label;
   } while(1);

   return electrodeFromGid;
}

// Basic inline progress reporting

double ecg_readParGF(OBJECT* obj, const std::string VmFilename, const int global_size, const std::unordered_map<int,int> &gfFromGid, std::vector<double>& final_values) {
   // Reconstruct same MPI basics as outside the function
   int num_ranks, my_rank;
   MPI_Comm_size(MPI_COMM_WORLD,&num_ranks);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   // Store pio input in my_*, use MPI to give everybody combined_*
   std::vector<int> my_keys, combined_keys;
   std::vector<double> my_values, combined_values;

   // Actual pio input handle
   PFILE* file = Popen(VmFilename.c_str(), "r", MPI_COMM_WORLD);

   // Read metadata for time step
   OBJECT* hObj = file->headerObject;
   std::vector<std::string> fieldNames,  fieldTypes;
   objectGet(hObj, "field_names", fieldNames); // field_names = gid Vm dVmD dVmR;
   objectGet(hObj, "field_types", fieldTypes); // field_types = u f f f;
   double time;
   objectGet(hObj, "time", time, "-1");
   assert(file->datatype == FIXRECORDASCII);

   // Prepare to read in random inputs in a random order
   int my_num_inputs = 0;
   std::map<int,double> my_inputs;

   // Main input engine
   PIO_FIXED_RECORD_HELPER* helper = (PIO_FIXED_RECORD_HELPER*) file->helper;
   unsigned lrec = helper->lrec;
   unsigned nRecords = file->bufsize/lrec;
   for (unsigned int irec=0; irec<nRecords; irec++) { //read in the file
      unsigned maxRec = 2048;          // Maximum record *length*
      char buf[maxRec+1];
      
      Pfgets(buf, maxRec, file);
      assert(strlen(buf) < maxRec);

      std::stringstream readline(buf);

      int gid;   readline >> gid; // For now we happen to know gid is column 1
      double Vm; readline >> Vm;  // For now we happen to know data is column 2

      assert(gfFromGid.find(gid) != gfFromGid.end());
      my_inputs[gfFromGid.find(gid)->second] = Vm;
      my_num_inputs++;
   }

   // Share input buffer sizes
   std::vector<int> num_inputs_per_rank(num_ranks);
   MPI_Allgather(&my_num_inputs,1,MPI_INT,
		 num_inputs_per_rank.data(),1,MPI_INT,MPI_COMM_WORLD);

   // Share actual inputs
   std::vector<int> rank_offsets(num_ranks);
   {
      int next_offset = 0;
      for(int i=0; i<num_ranks; i++) {
	 rank_offsets[i] = next_offset;
	 next_offset += num_inputs_per_rank[i];
      }
   }
#ifdef DEBUG
   std::cout << "Local size " << my_num_inputs
	     << ", global size " << global_size << "." << std::endl;
#endif

   // Prepare buffers for MPI sync
   my_keys.resize(my_num_inputs);
   combined_keys.resize(global_size);
   my_values.resize(my_num_inputs);
   combined_values.resize(global_size);
   final_values.resize(global_size);

   // Flatten random map to a pair of contiguous vectors
   {
      int i=0;
      for(std::map<int,double>::iterator it=my_inputs.begin();
	  it!=my_inputs.end(); ++it) {
	 my_keys[i] = it->first;
	 my_values[i] = it->second;
	 i++;
      }
   }

   // Merge all pio data into contiguous regions
   //  (on all ranks, at least for now)
   int my_offset = rank_offsets[my_rank];
   MPI_Allgatherv(my_keys.data(),my_num_inputs,MPI_INT,
		  combined_keys.data(),num_inputs_per_rank.data(),
		  rank_offsets.data(),MPI_INT,MPI_COMM_WORLD);
   MPI_Allgatherv(my_values.data(),my_num_inputs,MPI_DOUBLE,
		  combined_values.data(),num_inputs_per_rank.data(),
		  rank_offsets.data(),MPI_DOUBLE,MPI_COMM_WORLD);

   // Sort combined data based on keys
   for(int i=0; i<global_size; i++) {
      final_values[combined_keys[i]] = combined_values[i];
   }

   // Clean up and terminate
   Pclose(file);
   return time;
}
