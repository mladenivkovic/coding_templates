#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "ParticleIdentifier.h"
#include "tarch/la/Vector.h"
// #include "TestParticle.h"

// #include "Database.h"


/**
 * Test whether ParticleIdentifier and ParticleSearchIdentifier
 * work as intended in a map for a fuzzy search.
 */
void testTruthTableSearchAndIDKeys(){

  namespace ac = ::toolbox::particles::assignmentchecks;

  tarch::la::Vector<Dimensions, double> particleAX;
  tarch::la::Vector<Dimensions, double> particleBX;

  for (int i = 0; i < Dimensions; i++){
    particleAX(i) = 1.;
    particleBX(i) = 2.;
  }

  // create a key and assign it a value in the map
  ac::internal::ParticleIdentifier a = ac::internal::ParticleIdentifier("DummyParticle", particleAX, 1);
  ac::internal::ParticleIdentifier b = ac::internal::ParticleIdentifier("DummyParticle", particleBX, 2);


  // std::cout << "a < b:" << (a < b) << "(1) a==b:" << (a == b) <<
  //   "(0) a==a:" <<  (a == a) << "(1) b==b:" << (b == b) << "(1)" << std::endl;

  assert(a < b);
  assert(not (a==b));
  assert(not (b==a));
  assert(not (b < a));

  const double tol = 0.5;
  ac::internal::ParticleSearchIdentifier A = ac::internal::ParticleSearchIdentifier("DummyParticle", particleAX, 1, tol);
  ac::internal::ParticleSearchIdentifier B = ac::internal::ParticleSearchIdentifier("DummyParticle", particleBX, 2, tol);

  // std::cout << "A < B:" << (A < B) << "(1) A==B:" << (A == B)
  //   << "(0) A==A:" <<  (A == A) << "(1) B==B:" << (B == B) << "(1)" << std::endl;

  assert(A < B);
  assert(not (A==B));
  assert(not (B==A));
  assert(not (B < A));


  // cross-comparisons
  // std::cout << "a < A:" << (a < A) << "(0) A < a:" << (A < a) << "(0) a==A:"
  //   << (a == A) << "(1) A == a:" <<  (A == a) << "(1)" << std::endl;
  // std::cout << "a < B:" << (a < B) << "(1) B < a:" << (B < a) << "(0) a==B:"
  //   << (a == B) << "(0) B == a:" <<  (B == a) << "(0)" << std::endl;
  // std::cout << "b < A:" << (b < A) << "(0) A < b:" << (A < b) << "(1) b==A:"
  //   << (b == A) << "(0) A == b:" <<  (A == b) << "(0)" << std::endl;
  // std::cout << "b < B:" << (b < B) << "(0) B < b:" << (B < b) << "(0) b==B:"
  //   << (b == B) << "(1) B == b:" <<  (B == b) << "(1)" << std::endl;

  assert(not (a < A));
  assert(not (A < a));
  assert(A == a);
  assert(a == A);

  assert(a < B);
  assert(not (B < a));
  assert(not (B == a));
  assert(not (a == B));

  assert(not (b < A));
  assert(A < b);
  assert(not (A == b));
  assert(not (b == A));

  assert(not (b < B));
  assert(not (B < b));
  assert(b == B);
  assert(B == b);

}



/*  */
/*  */
/*  */
/*  */
/**
 * Make sure that adding mesh sweeps to the database works.
 */
/* void test_adding_sweeps_to_database(void){ */
/*  */
/*   namespace ac = ::toolbox::particles::assignmentchecks; */
/*  */
/*   ac::internal::Database& eventDatabase = ac::getDatabaseInstance(); */
/*   eventDatabase.reset(); */
/*   ac::ensureDatabaseIsEmpty(); */
/*  */
/*  */
/*   std::vector<std::string> meshSweepNames = {"initial", "alpha", "beta", "gamma", "delta"}; */
/*  */
/*  */
/*   std::vector<std::string>::iterator sweep = meshSweepNames.begin(); */
/*   // Skip first, which gets automatically added to the database. */
/*   sweep++; */
/*  */
/*   while (sweep != meshSweepNames.end()){ */
/*     ac::startMeshSweep(*sweep); */
/*     sweep++; */
/*   } */
/*  */
/*   // Check number of sweeps correct */
/*   assertion3(eventDatabase.getMeshSweepData().size() == meshSweepNames.size(), */
/*       "Wrong number of mesh sweeps in database", */
/*       eventDatabase.getMeshSweepData().size(), */
/*       meshSweepNames.size() */
/*       ); */
/*  */
/*   // Check our bookkeeping */
/*   assertion3((eventDatabase.getCurrentMeshSweepIndex() + 1) == meshSweepNames.size(), */
/*       "Wrong count of mesh sweeps in database", */
/*       eventDatabase.getMeshSweepData().size(), */
/*       meshSweepNames.size() */
/*       ); */
/*  */
/*  */
/*   auto databaseSweep = eventDatabase.getMeshSweepData().begin(); */
/*   sweep = meshSweepNames.begin(); */
/*  */
/*   while (sweep != meshSweepNames.end()){ */
/*  */
/*     assertion3(*sweep == databaseSweep->getName(), "Wrong Sweep Name", *sweep, databaseSweep->getName()); */
/*     sweep++; */
/*     databaseSweep++; */
/*   } */
/*  */
/*   // Clean up after yourself. */
/*   eventDatabase.reset(); */
/*   ac::ensureDatabaseIsEmpty(); */
/* } */
/*  */
/*  */
/*  */
/**
 * Test the adding of particle events to the database.
 * This also serves as a unit test for all possible events: I let
 * the particles run through some meaningless mesh sweeps, and add
 * all the event types.
 *
 * Here, I only use static particles, so the identifier remains
 * the same all the time. Switching particle identifiers will be
 * done in a separate test.
 * We're also adding events without them having any meaning. Proper
 * event tracing including consistency checks will also be done later.
 */
/* void test_adding_particle_events(bool verbose=false){ */
/*  */
/*   namespace ac = ::toolbox::particles::assignmentchecks; */
/*  */
/*   ac::internal::Database& eventDatabase = ac::getDatabaseInstance(); */
/*   eventDatabase.reset(); */
/*   ac::ensureDatabaseIsEmpty(); */
/*   // make sure we're not deleting anything just yet. */
/*   // re-initialize with enough "space". */
/*   eventDatabase = ac::internal::Database(100); */
/*  */
/*   int treeId = 1; */
/*   bool isLocal = true; */
/*   double positionTolerance = 1.; */
/*   tarch::la::Vector<Dimensions, double> particleX; */
/*   tarch::la::Vector<Dimensions, double> vertexX; */
/*   tarch::la::Vector<Dimensions, double> vertexH; */
/*  */
/*   std::string trace; */
/*  */
/*   std::vector<std::string> meshSweepNames = {"alpha", "beta", "gamma", "delta"}; */
/*  */
/*  */
/*   int nparts = 10; */
/*   std::vector<std::string>::iterator sweep = meshSweepNames.begin(); */
/*   while (sweep != meshSweepNames.end()){ */
/*     ac::startMeshSweep(*sweep); */
/*  */
/*     for (int p = 1; p < nparts+1; p++){ */
/*  */
/*       int particleID = p; */
/*  */
/*       for (int i = 0; i < Dimensions; i++){ */
/*         particleX(i) = p; */
/*         vertexX(i) = p; */
/*         vertexH(i) = p; */
/*       } */
/*  */
/*       ac::internal::ParticleIdentifier identifier = ac::internal::ParticleIdentifier( */
/*           "DummyParticle", */
/*           particleX, */
/*           particleID, */
/*           positionTolerance */
/*       ); */
/*  */
/*       // vertex assignment event */
/*       trace = "Assign/sweep:" + *sweep; */
/*       ac::internal::Event vassEvent = ac::internal::Event( */
/*           ac::internal::Event::Type::AssignToVertex, */
/*           isLocal, */
/*           vertexX, */
/*           particleX, */
/*           vertexH, */
/*           treeId, */
/*           trace); */
/*  */
/*       eventDatabase.addEvent(identifier, vassEvent); */
/*  */
/*       // move event */
/*       trace = "Move/sweep:" + *sweep; */
/*       ac::internal::Event moveEvent = ac::internal::Event( */
/*           ac::internal::Event::Type::MoveWhileAssociatedToVertex, */
/*           vertexX, */
/*           particleX, */
/*           vertexH, */
/*           treeId, */
/*           trace); */
/*       eventDatabase.addEvent(identifier, moveEvent); */
/*  */
/*       // sieve event */
/*       trace = "Sieve/sweep:" + *sweep; */
/*       ac::internal::Event sievEvent = ac::internal::Event( */
/*           ac::internal::Event::Type::AssignToSieveSet, */
/*           isLocal, */
/*           treeId, */
/*           trace); */
/*       eventDatabase.addEvent(identifier, sievEvent); */
/*  */
/*       // invalid event */
/*       trace = "NotFound/sweep:" + *sweep; */
/*       ac::internal::Event invalidEvent = ac::internal::Event( */
/*           ac::internal::Event::Type::NotFound); */
/*       eventDatabase.addEvent(identifier, invalidEvent); */
/*  */
/*     } */
/*  */
/*     sweep++; */
/*   } */
/*  */
/*  */
/*   if (verbose) { */
/*     // Print out particle histories */
/*     for (int p = 1; p < nparts+1; p++){ */
/*  */
/*       int particleID = p; */
/*  */
/*       for (int i = 0; i < Dimensions; i++){ */
/*         particleX(i) = p; */
/*         vertexX(i) = p; */
/*         vertexH(i) = p; */
/*       } */
/*  */
/*       ac::internal::ParticleIdentifier identifier = ac::internal::ParticleIdentifier( */
/*           "DummyParticle", */
/*           particleX, */
/*           particleID, */
/*           positionTolerance */
/*       ); */
/*  */
/*       std::cout << eventDatabase.particleHistory(identifier); */
/*     } */
/*   } */
/*  */
/*   // Clean up after yourself. */
/*   eventDatabase.reset(); */
/*   ac::ensureDatabaseIsEmpty(); */
/* } */
/*  */
/*  */
/**
 * Test the adding of particle events to the database with a moving
 * particle. The catch is that a moving particle will eventually
 * need to change its identifier in the database.
 *
 * We're also adding events without them having any meaning. Proper
 * event tracing including consistency checks will also be done later.
 */
/* void test_adding_moving_particle_events(bool verbose=false){ */
/*  */
/*   namespace ac = ::toolbox::particles::assignmentchecks; */
/*  */
/*   ac::internal::Database& eventDatabase = ac::getDatabaseInstance(); */
/*   eventDatabase.reset(); */
/*   ac::ensureDatabaseIsEmpty(); */
/*   // make sure we're not deleting anything just yet. */
/*   // re-initialize with enough "space". */
/*   eventDatabase = ac::internal::Database(100); */
/*  */
/*   int particleID = 1; */
/*   int treeId = 1; */
/*   // bool isLocal = true; */
/*   double positionTolerance = 1.; */
/*   double dt = 1.; */
/*  */
/*   tarch::la::Vector<Dimensions, double> particleX; */
/*   tarch::la::Vector<Dimensions, double> vertexX; */
/*   tarch::la::Vector<Dimensions, double> vertexH; */
/*  */
/*   // particle displacement each step */
/*   tarch::la::Vector<Dimensions, double> dx; */
/*  */
/*   for (int i = 0; i < Dimensions; i++){ */
/*     particleX(i) = 1.; */
/*     vertexX(i) = 1.; */
/*     vertexH(i) = 1.; */
/*     dx(i) = 0.0005; */
/*   } */
/*  */
/*  */
/*  */
/*   for (int sweep = 0; sweep < 100; sweep++){ */
/*  */
/*     std::string sweepname = "Sweep" + std::to_string(sweep); */
/*     ac::startMeshSweep(sweepname); */
/*  */
/*     // move particle */
/*     for (int i = 0; i < Dimensions; i++){ */
/*       particleX(i) += dx(i) * dt; */
/*     } */
/*  */
/*     // generate new identifier */
/*     ac::internal::ParticleIdentifier identifier = ac::internal::ParticleIdentifier( */
/*         "DummyParticle", */
/*         particleX, */
/*         particleID, */
/*         positionTolerance */
/*     ); */
/*  */
/*  */
/*     // move event */
/*     std::string trace = "Move/sweep:" + std::to_string(sweep); */
/*     ac::internal::Event moveEvent = ac::internal::Event( */
/*         ac::internal::Event::Type::MoveWhileAssociatedToVertex, */
/*         vertexX, */
/*         particleX, */
/*         vertexH, */
/*         treeId, */
/*         trace); */
/*     eventDatabase.addEvent(identifier, moveEvent); */
/*  */
/*   } */
/*  */
/*  */
/*   if (verbose) { */
/*     // Print out particle histories */
/*  */
/*     ac::internal::ParticleIdentifier identifier = ac::internal::ParticleIdentifier( */
/*         "DummyParticle", */
/*         particleX, */
/*         particleID, */
/*         positionTolerance */
/*     ); */
/*  */
/*     std::cout << eventDatabase.particleHistory(identifier); */
/*   } */
/*  */
/*  */
/*   std::cout << "DATABASE DUMP\n\n" << eventDatabase.toString(); */
/*  */
/*  */
/*   // Clean up after yourself. */
/*   eventDatabase.reset(); */
/*   ac::ensureDatabaseIsEmpty(); */
/* } */
/*  */
/*  */




int main(void) {

  bool verbose = false;
  testTruthTableSearchAndIDKeys();
  // test_adding_sweeps_to_database();
  // test_adding_particle_events(false);
  // test_adding_moving_particle_events(true);

  std::cout << "Done. Bye!" <<  std::endl;

}
