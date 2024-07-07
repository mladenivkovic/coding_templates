#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "ParticleIdentifier.h"
#include "tarch/la/Vector.h"
// #include "TestParticle.h"

#include "Database.h"

#include "TestHelpers.h"




int main(void) {

  bool verbose = true;

  toolbox::particles::assignmentchecks::tests::TestHelpers runner = toolbox::particles::assignmentchecks::tests::TestHelpers();
  runner.testTruthTableSearchAndIDKeys(verbose);
  runner.testAddingSweepsToDatabase(verbose);
  runner.testAddingParticleEvents(verbose);
  // test shifting particle identifier without trimming database
  runner.testAddingParticleMovingEvents(100, 1000, verbose);
  // test shifting particle identifier and trimming database
  runner.testAddingParticleMovingEvents(100, 16, verbose);


  runner.testParticleWalkSameTreeLevel();
  runner.testParticleLiftDrop();
  runner.testParticleWalk();
  runner.testPeriodicBoundaryConditions();

  runner.testSieveSet();

  std::cout << "Done. Bye!" <<  std::endl;

}
