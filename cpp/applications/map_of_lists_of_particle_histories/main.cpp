#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "tarch/la/Vector.h"
#include "TestParticle.h"

#include "Database.h"






/**
 * Make sure that adding mesh sweeps to the database works.
 */
void test_adding_sweeps_to_database(void){


  namespace as = ::toolbox::particles::assignmentchecks;

  as::internal::Database& eventDatabase = as::getDatabaseInstance();

  eventDatabase.reset();
  as::ensureDatabaseIsEmpty();


  std::vector<std::string> meshSweepNames = {"initial", "alpha", "beta", "gamma", "delta"};


  std::vector<std::string>::iterator sweep = meshSweepNames.begin();
  // Skip first, which gets automatically added to the database.
  sweep++;

  while (sweep != meshSweepNames.end()){
    as::startMeshSweep(*sweep);
    sweep++;
  }

  // Check number of sweeps correct
  assertion3(eventDatabase.getMeshSweepData().size() == meshSweepNames.size(),
      "Wrong number of mesh sweeps in database",
      eventDatabase.getMeshSweepData().size(),
      meshSweepNames.size()
      );

  // Check our bookkeeping
  assertion3((eventDatabase.getCurrentMeshSweepIndex() + 1) == meshSweepNames.size(),
      "Wrong count of mesh sweeps in database",
      eventDatabase.getMeshSweepData().size(),
      meshSweepNames.size()
      );


  auto databaseSweep = eventDatabase.getMeshSweepData().begin();
  sweep = meshSweepNames.begin();

  while (sweep != meshSweepNames.end()){

    assertion3(*sweep == databaseSweep->getName(), "Wrong Sweep Name", *sweep, databaseSweep->getName());
    sweep++;
    databaseSweep++;
  }

  // Clean up after yourself.
  eventDatabase.reset();
  as::ensureDatabaseIsEmpty();
}






int main(void) {

  namespace as = ::toolbox::particles::assignmentchecks;

  std::cout << "Hello There!" <<  std::endl;

  test_adding_sweeps_to_database();


}
