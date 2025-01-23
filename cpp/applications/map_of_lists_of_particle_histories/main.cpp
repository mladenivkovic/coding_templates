#include "TestHelpers.h"
#include <iostream>

int main(void) {

  bool verbose = true;

  toolbox::particles::assignmentchecks::tests::TestHelpers runner =
      toolbox::particles::assignmentchecks::tests::TestHelpers();
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

  std::cout << "Done. Bye!" << std::endl;
}
