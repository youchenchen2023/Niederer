# hack

   virtual void getCheckpointInfo(std::vector<std::string>& fieldNames,
                                  std::vector<std::string>& fieldUnits) const
                                  {
                                    fieldNames.clear();
                                    fieldUnits.clear();
                                  };
   virtual int getVarHandle(const std::string& varName) const
                                  {
                                    return -1;
                                  };
