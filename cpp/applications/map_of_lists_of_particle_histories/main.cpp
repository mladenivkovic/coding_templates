#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "ParticleIdentifier.h"
#include "tarch/la/Vector.h"
#include "TestParticle.h"

#include "Database.h"






/**
 * Make sure that adding mesh sweeps to the database works.
 */
void test_adding_sweeps_to_database(void){

  namespace ac = ::toolbox::particles::assignmentchecks;

  ac::internal::Database& eventDatabase = ac::getDatabaseInstance();
  eventDatabase.reset();
  ac::ensureDatabaseIsEmpty();


  std::vector<std::string> meshSweepNames = {"initial", "alpha", "beta", "gamma", "delta"};


  std::vector<std::string>::iterator sweep = meshSweepNames.begin();
  // Skip first, which gets automatically added to the database.
  sweep++;

  while (sweep != meshSweepNames.end()){
    ac::startMeshSweep(*sweep);
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
  ac::ensureDatabaseIsEmpty();
}



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
void test_adding_particle_events(bool verbose=false){

  namespace ac = ::toolbox::particles::assignmentchecks;

  ac::internal::Database& eventDatabase = ac::getDatabaseInstance();
  eventDatabase.reset();
  ac::ensureDatabaseIsEmpty();
  // make sure we're not deleting anything just yet.
  // re-initialize with enough "space".
  eventDatabase = ac::internal::Database(100);

  int treeId = 1;
  bool isLocal = true;
  double positionTolerance = 1.;
  tarch::la::Vector<Dimensions, double> particleX;
  tarch::la::Vector<Dimensions, double> vertexX;
  tarch::la::Vector<Dimensions, double> vertexH;

  std::string trace;

  std::vector<std::string> meshSweepNames = {"alpha", "beta", "gamma", "delta"};


  int nparts = 10;
  std::vector<std::string>::iterator sweep = meshSweepNames.begin();
  while (sweep != meshSweepNames.end()){
    ac::startMeshSweep(*sweep);

    for (int p = 1; p < nparts+1; p++){

      int particleID = p;

      for (int i = 0; i < Dimensions; i++){
        particleX(i) = p;
        vertexX(i) = p;
        vertexH(i) = p;
      }

      ac::internal::ParticleIdentifier identifier = ac::internal::ParticleIdentifier(
          "DummyParticle",
          particleX,
          particleID,
          positionTolerance
      );

      // vertex assignment event
      trace = "Assign/sweep:" + *sweep;
      ac::internal::Event vassEvent = ac::internal::Event(
          ac::internal::Event::Type::AssignToVertex,
          isLocal,
          vertexX,
          particleX,
          vertexH,
          treeId,
          trace);

      eventDatabase.addEvent(identifier, vassEvent);

      // move event
      trace = "Move/sweep:" + *sweep;
      ac::internal::Event moveEvent = ac::internal::Event(
          ac::internal::Event::Type::MoveWhileAssociatedToVertex,
          vertexX,
          particleX,
          vertexH,
          treeId,
          trace);
      eventDatabase.addEvent(identifier, moveEvent);

      // sieve event
      trace = "Sieve/sweep:" + *sweep;
      ac::internal::Event sievEvent = ac::internal::Event(
          ac::internal::Event::Type::AssignToSieveSet,
          isLocal,
          treeId,
          trace);
      eventDatabase.addEvent(identifier, sievEvent);

      // invalid event
      trace = "NotFound/sweep:" + *sweep;
      ac::internal::Event invalidEvent = ac::internal::Event(
          ac::internal::Event::Type::NotFound);
      eventDatabase.addEvent(identifier, invalidEvent);

    }

    sweep++;
  }


  if (not verbose) return;

  // Print out particle histories
  for (int p = 1; p < nparts+1; p++){

    int particleID = p;

    for (int i = 0; i < Dimensions; i++){
      particleX(i) = p;
      vertexX(i) = p;
      vertexH(i) = p;
    }

    ac::internal::ParticleIdentifier identifier = ac::internal::ParticleIdentifier(
        "DummyParticle",
        particleX,
        particleID,
        positionTolerance
    );

    std::cout << eventDatabase.particleHistory(identifier);
  }

  // Clean up after yourself.
  eventDatabase.reset();
  ac::ensureDatabaseIsEmpty();
}


/**
 * event tracing including consistency checks will also be done later.
 */
void test_adding_moving_particle_events(void){

  namespace ac = ::toolbox::particles::assignmentchecks;

  ac::internal::Database& eventDatabase = ac::getDatabaseInstance();
  eventDatabase.reset();
  ac::ensureDatabaseIsEmpty();

  int particleID = 1;
  int treeId = 1;
  bool isLocal = true;
  double positionTolerance = 1.;
  tarch::la::Vector<Dimensions, double> particleX;
  tarch::la::Vector<Dimensions, double> vertexX;
  tarch::la::Vector<Dimensions, double> vertexH;
  for (int i = 0; i < Dimensions; i++){
    particleX(i) = 1.;
    vertexX(i) = 1.;
    vertexH(i) = 1.;
  }

  ac::internal::ParticleIdentifier identifier = ac::internal::ParticleIdentifier(
      "DummyParticle",
      particleX,
      particleID,
      positionTolerance
  );

  std::string trace;

  std::vector<std::string> meshSweepNames = {"alpha", "beta", "gamma", "delta"};
  std::vector<std::string>::iterator sweep = meshSweepNames.begin();


  int nparts = 10;

  for (int p = 0; p < nparts; p++){

  }

  while (sweep != meshSweepNames.end()){
    ac::startMeshSweep(*sweep);


    // vertex assignment event
    trace = "Assign/sweep:" + *sweep;
    ac::internal::Event vassEvent = ac::internal::Event(
        ac::internal::Event::Type::AssignToVertex,
        isLocal,
        vertexX,
        particleX,
        vertexH,
        treeId,
        trace);

    eventDatabase.addEvent(identifier, vassEvent);

    // move event
    trace = "Move/sweep:" + *sweep;
    ac::internal::Event moveEvent = ac::internal::Event(
        ac::internal::Event::Type::MoveWhileAssociatedToVertex,
        vertexX,
        particleX,
        vertexH,
        treeId,
        trace);
    eventDatabase.addEvent(identifier, moveEvent);

    // sieve event
    trace = "Sieve/sweep:" + *sweep;
    ac::internal::Event sievEvent = ac::internal::Event(
        ac::internal::Event::Type::AssignToSieveSet,
        isLocal,
        treeId,
        trace);
    eventDatabase.addEvent(identifier, sievEvent);

    // invalid event
    trace = "NotFound/sweep:" + *sweep;
    ac::internal::Event invalidEvent = ac::internal::Event(
        ac::internal::Event::Type::NotFound);
    eventDatabase.addEvent(identifier, invalidEvent);

    sweep++;
  }




  // Clean up after yourself.
  eventDatabase.reset();
  ac::ensureDatabaseIsEmpty();
}






int main(void) {

  bool verbose = false;
  test_adding_sweeps_to_database();
  test_adding_particle_events(verbose);

  std::cout << "Done. Bye!" <<  std::endl;
}
