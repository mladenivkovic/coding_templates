// This file is part of the Peano project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org

#include "TestHelpers.h"
#include "Database.h"
#include "TestParticle.h"
#include "TracingAPI.h"

// TODO: Clean up again
// #include "toolbox/particles/assignmentchecks/TracingAPI.h"
// #include "toolbox/particles/assignmentchecks/Utils.h"
// #include "Utils.h"

// TODO: All the "Verbose" parameters everywhere
// TODO: Add actual  assertions everywhere

#ifdef UseTestSpecificCompilerSettings
#pragma optimize("", off)
#endif

// TODO: Put back in
// toolbox::particles::assignmentchecks::tests::TestHelpers::TestHelpers():
//   TestCase("toolbox::particles::assignmentchecks::tests::TestHelpers") {}
toolbox::particles::assignmentchecks::tests::TestHelpers::TestHelpers() {};

bool toolbox::particles::assignmentchecks::tests::internal::liftParticle(
    const tarch::la::Vector<Dimensions, double> particleX,
    const tarch::la::Vector<Dimensions, double> vertexX,
    const tarch::la::Vector<Dimensions, double> vertexH, const int depth) {

  if (depth == 0)
    return false;

  tarch::la::Vector<Dimensions, double> dv = vertexX - particleX;

  if (tarch::la::oneSmaller(dv, -1. * vertexH))
    return true;
  if (tarch::la::oneGreater(dv, vertexH))
    return true;
  return false;
}

bool toolbox::particles::assignmentchecks::tests::internal::dropParticle(
    const tarch::la::Vector<Dimensions, double> particleX,
    const tarch::la::Vector<Dimensions, double> vertexH, const int depth,
    const int maxVertexDepth) {

  if (depth == maxVertexDepth)
    return false;

  tarch::la::Vector<Dimensions, double> vHnew = vertexH / 3.;
  tarch::la::Vector<Dimensions, double> vertexX = findVertexX(particleX, vHnew);

  tarch::la::Vector<Dimensions, double> dv = vertexX - particleX;

  if (tarch::la::allSmaller(dv, vHnew) and
      tarch::la::allGreater(dv, -1. * vHnew))
    return true;
  return false;
}

int toolbox::particles::assignmentchecks::tests::internal::findVertexInd(
    double x, double vertexH) {

  // vertexH is half vertex square size. So multiply by 2 here.
  int vertexInd = static_cast<int>(std::floor((x / (2. * vertexH)) + 0.5));

  return vertexInd;
}

tarch::la::Vector<Dimensions, double>
toolbox::particles::assignmentchecks::tests::internal::findVertexX(
    const tarch::la::Vector<Dimensions, double> x,
    const tarch::la::Vector<Dimensions, double> vertexH) {
  tarch::la::Vector<Dimensions, double> vertexX;

  for (int i = 0; i < Dimensions; i++) {
    int vertexInd = internal::findVertexInd(x(i), vertexH(i));
    vertexX(i) = 2 * vertexInd * vertexH(i);
  }

  return vertexX;
}

void toolbox::particles::assignmentchecks::tests::TestHelpers::
    testTruthTableSearchAndIDKeys(bool verbose) {

  if (verbose) {
    std::cout << "------------------------------------------\n"
              << "Running testTruthTableSearchAndIDKeys()\n"
              << "------------------------------------------\n"
              << std::endl;
  }

  namespace ac = ::toolbox::particles::assignmentchecks;

  tarch::la::Vector<Dimensions, double> particleAX;
  tarch::la::Vector<Dimensions, double> particleBX;

  for (int i = 0; i < Dimensions; i++) {
    particleAX(i) = 1.;
    particleBX(i) = 2.;
  }

  // create a key and assign it a value in the map
  ac::internal::ParticleIdentifier a =
      ac::internal::ParticleIdentifier("DummyParticle", particleAX, 1);
  ac::internal::ParticleIdentifier b =
      ac::internal::ParticleIdentifier("DummyParticle", particleBX, 2);

  // std::cout << "a < b:" << (a < b) << "(1) a==b:" << (a == b) <<
  //   "(0) a==a:" <<  (a == a) << "(1) b==b:" << (b == b) << "(1)" <<
  //   std::endl;

  assert(a < b);
  assert(not(a == b));
  assert(not(b == a));
  assert(not(b < a));

  const double tol = 0.5;
  ac::internal::ParticleSearchIdentifier A =
      ac::internal::ParticleSearchIdentifier("DummyParticle", particleAX, 1,
                                             tol);
  ac::internal::ParticleSearchIdentifier B =
      ac::internal::ParticleSearchIdentifier("DummyParticle", particleBX, 2,
                                             tol);

  // std::cout << "A < B:" << (A < B) << "(1) A==B:" << (A == B)
  //   << "(0) A==A:" <<  (A == A) << "(1) B==B:" << (B == B) << "(1)" <<
  //   std::endl;

  assert(A < B);
  assert(not(A == B));
  assert(not(B == A));
  assert(not(B < A));

  // cross-comparisons
  // std::cout << "a < A:" << (a < A) << "(0) A < a:" << (A < a) << "(0) a==A:"
  //   << (a == A) << "(1) A == a:" <<  (A == a) << "(1)" << std::endl;
  // std::cout << "a < B:" << (a < B) << "(1) B < a:" << (B < a) << "(0) a==B:"
  //   << (a == B) << "(0) B == a:" <<  (B == a) << "(0)" << std::endl;
  // std::cout << "b < A:" << (b < A) << "(0) A < b:" << (A < b) << "(1) b==A:"
  //   << (b == A) << "(0) A == b:" <<  (A == b) << "(0)" << std::endl;
  // std::cout << "b < B:" << (b < B) << "(0) B < b:" << (B < b) << "(0) b==B:"
  //   << (b == B) << "(1) B == b:" <<  (B == b) << "(1)" << std::endl;

  assert(not(a < A));
  assert(not(A < a));
  assert(A == a);
  assert(a == A);

  assert(a < B);
  assert(not(B < a));
  assert(not(B == a));
  assert(not(a == B));

  assert(not(b < A));
  assert(A < b);
  assert(not(A == b));
  assert(not(b == A));

  assert(not(b < B));
  assert(not(B < b));
  assert(b == B);
  assert(B == b);
}

void toolbox::particles::assignmentchecks::tests::TestHelpers::
    testAddingSweepsToDatabase(bool verbose) {

  if (verbose) {
    std::cout << "------------------------------------------\n"
              << "Running testAddingSweepsToDatabase()\n"
              << "------------------------------------------\n"
              << std::endl;
  }

  namespace ac = ::toolbox::particles::assignmentchecks;

  ac::internal::Database &eventDatabase = ac::internal::Database::getInstance();
  eventDatabase.reset();
  ac::ensureDatabaseIsEmpty();

  std::vector<std::string> meshSweepNames = {"initial", "alpha", "beta",
                                             "gamma", "delta"};

  std::vector<std::string>::iterator sweep = meshSweepNames.begin();
  // Skip first, which gets automatically added to the database.
  sweep++;

  while (sweep != meshSweepNames.end()) {
    ac::startMeshSweep(*sweep);
    sweep++;
  }

  // Check number of sweeps correct
  assertion4(eventDatabase.getMeshSweepData().size() == meshSweepNames.size(),
             "Wrong number of mesh sweeps in database",
             eventDatabase.getMeshSweepData().size(), meshSweepNames.size(),
             eventDatabase.toString());

  // Check our bookkeeping
  assertion4((eventDatabase.getCurrentMeshSweepIndex() + 1) ==
                 meshSweepNames.size(),
             "Wrong count of mesh sweeps in database",
             eventDatabase.getMeshSweepData().size(), meshSweepNames.size(),
             eventDatabase.toString());

  // Check that names have been stored correctly

  auto databaseSweep = eventDatabase.getMeshSweepData().begin();
  sweep = meshSweepNames.begin();

  while (sweep != meshSweepNames.end()) {

    assertion3(*sweep == databaseSweep->getName(), "Wrong Sweep Name", *sweep,
               databaseSweep->getName());
    sweep++;
    databaseSweep++;
  }

  // Clean up after yourself.
  eventDatabase.reset();
  ac::ensureDatabaseIsEmpty();
}

void toolbox::particles::assignmentchecks::tests::TestHelpers::
    testAddingParticleEvents(bool verbose) {

  if (verbose) {
    std::cout << "------------------------------------------\n"
              << "Running testAddingParticleEvents()\n"
              << "------------------------------------------\n"
              << std::endl;
  }

  namespace ac = ::toolbox::particles::assignmentchecks;

  ac::internal::Database &eventDatabase = ac::internal::Database::getInstance();
  eventDatabase.reset();
  ac::ensureDatabaseIsEmpty();
  // make sure we're not deleting anything just yet by having too many
  // snapshots. re-initialize with enough "space".
  eventDatabase.setMaxParticleSnapshotsToKeepTrackOf(100);

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
  while (sweep != meshSweepNames.end()) {
    ac::startMeshSweep(*sweep);

    for (int p = 1; p < nparts + 1; p++) {

      int particleID = p;

      for (int i = 0; i < Dimensions; i++) {
        particleX(i) = p;
        vertexX(i) = p;
        vertexH(i) = p;
      }

      ac::internal::ParticleSearchIdentifier identifier =
          ac::internal::ParticleSearchIdentifier("DummyParticle", particleX,
                                                 particleID, positionTolerance);

      // vertex assignment event
      trace = "Assign/sweep:" + *sweep;
      ac::internal::Event vassEvent = ac::internal::Event(
          ac::internal::Event::Type::AssignToVertex, isLocal, vertexX,
          particleX, vertexH, treeId, trace);

      eventDatabase.addEvent(identifier, vassEvent);

      // move event
      trace = "Move/sweep:" + *sweep;
      ac::internal::Event moveEvent = ac::internal::Event(
          ac::internal::Event::Type::MoveWhileAssociatedToVertex, vertexX,
          particleX, vertexH, treeId, trace);
      eventDatabase.addEvent(identifier, moveEvent);

      // sieve event
      trace = "Sieve/sweep:" + *sweep;
      ac::internal::Event sievEvent = ac::internal::Event(
          ac::internal::Event::Type::AssignToSieveSet, isLocal, treeId, trace);
      eventDatabase.addEvent(identifier, sievEvent);

      // invalid event
      trace = "NotFound/sweep:" + *sweep;
      ac::internal::Event invalidEvent =
          ac::internal::Event(ac::internal::Event::Type::NotFound);
      eventDatabase.addEvent(identifier, invalidEvent);
    }

    sweep++;
  }

  // is particle count correct?
  assertion3(eventDatabase.getNumberOfTracedParticles() ==
                 static_cast<int>(nparts),
             "Wrong particle count in database",
             eventDatabase.getNumberOfTracedParticles(), nparts);

  // Is number of events per particle correct?
  for (int p = 1; p < nparts + 1; p++) {

    int particleID = p;

    for (int i = 0; i < Dimensions; i++) {
      particleX(i) = p;
      vertexX(i) = p;
      vertexH(i) = p;
    }

    ac::internal::ParticleSearchIdentifier identifier =
        ac::internal::ParticleSearchIdentifier("DummyParticle", particleX,
                                               particleID, positionTolerance);

    size_t nEntries = eventDatabase.getTotalParticleEntries(identifier);
    // we add 4 events per sweep.
    assertion4(nEntries == (meshSweepNames.size() * 4),
               "Wrong number of entries for particle ", p, nEntries,
               meshSweepNames.size() * 4);

    if (verbose) {
      // Print out particle histories
      std::cout << eventDatabase.particleHistory(identifier);
    }
  }

  // Clean up after yourself.
  eventDatabase.reset();
  ac::ensureDatabaseIsEmpty();
}

void toolbox::particles::assignmentchecks::tests::TestHelpers::
    testAddingParticleMovingEvents(int nsweeps, int nEventsToKeep,
                                   bool verbose) {

  if (verbose) {
    std::cout << "------------------------------------------\n"
              << "Running testAddingParticleMovingEvents()\n"
              << "------------------------------------------\n"
              << std::endl;
  }

  namespace ac = ::toolbox::particles::assignmentchecks;

  ac::internal::Database &eventDatabase = ac::internal::Database::getInstance();
  eventDatabase.reset();
  ac::ensureDatabaseIsEmpty();
  // make sure we're not deleting anything just yet.
  // re-initialize with enough "space".
  eventDatabase.setMaxParticleSnapshotsToKeepTrackOf(nEventsToKeep);

  int particleID = 1;
  int treeId = 1;
  double positionTolerance = 1.;
  double dt = 1.;

  tarch::la::Vector<Dimensions, double> particleX;
  tarch::la::Vector<Dimensions, double> vertexX;
  tarch::la::Vector<Dimensions, double> vertexH;

  // particle displacement each step
  tarch::la::Vector<Dimensions, double> dx;

  for (int i = 0; i < Dimensions; i++) {
    particleX(i) = 1.;
    vertexX(i) = 1.;
    vertexH(i) = 1.;
    dx(i) = 0.05;
  }

  // run through sweeps and add move events.
  for (int sweep = 0; sweep < nsweeps; sweep++) {

    std::string sweepname = "Sweep" + std::to_string(sweep);
    ac::startMeshSweep(sweepname);

    // move particle
    for (int i = 0; i < Dimensions; i++) {
      particleX(i) += dx(i) * dt;
    }

    // generate new identifier
    ac::internal::ParticleSearchIdentifier identifier =
        ac::internal::ParticleSearchIdentifier("DummyParticle", particleX,
                                               particleID, positionTolerance);

    // move event
    std::string trace = "Move/sweep:" + std::to_string(sweep);
    ac::internal::Event moveEvent = ac::internal::Event(
        ac::internal::Event::Type::MoveWhileAssociatedToVertex, vertexX,
        particleX, vertexH, treeId, trace);
    eventDatabase.addEvent(identifier, moveEvent);

    // Do we need to shift the identifier's coordinates?
    eventDatabase.shiftIdentifierCoordinates(identifier, particleX);
  }

  // is particle count correct?
  assertion2(eventDatabase.getNumberOfTracedParticles() == 1,
             "Wrong particle count in database, should be 1",
             eventDatabase.getNumberOfTracedParticles());

  // Did we record correct number of events?
  ac::internal::ParticleSearchIdentifier identifier =
      ac::internal::ParticleSearchIdentifier("DummyParticle", particleX,
                                             particleID, positionTolerance);

  int nEntries = eventDatabase.getTotalParticleEntries(identifier);
  // reduce nEventsToKeep by 1 here, as we always substitute the past
  // trajectory with the last event (and modify its trace to signify that)
  assertion6(nEntries == (nsweeps % (nEventsToKeep - 1)),
             "Wrong number of entries for particle ", nEntries, nsweeps,
             nEventsToKeep, nsweeps % (nEventsToKeep - 1),
             eventDatabase.particleHistory(identifier));

  if (verbose) {
    // Print out particle history
    std::cout << eventDatabase.particleHistory(identifier);
  }

  // Clean up after yourself.
  eventDatabase.reset();
  ac::ensureDatabaseIsEmpty();
}

void toolbox::particles::assignmentchecks::tests::TestHelpers::
    testParticleWalkSameTreeLevel() {

#if PeanoDebug > 0
  const bool verbose = false;

  namespace ac = toolbox::particles::assignmentchecks;

  if (verbose)
    std::cout << ">>>>>>>>>>>> RUNNING testParticleWalkSameTreeLevel() in "
              << Dimensions << "D" << std::endl;

  ac::internal::Database &eventDatabase = ac::internal::Database::getInstance();

  // IMPORTANT: Needs to be the same as the particle class name you use!
  // In traceParticleMovements(), this will be derived from the class name.
  std::string particleName = "TestParticle";

  // coordinates of particles
  tarch::la::Vector<Dimensions, double> localPartX;
  // particle displacement each step
  tarch::la::Vector<Dimensions, double> dx;
  // Assume we have vertices of a single size.
  tarch::la::Vector<Dimensions, double> vertexH;
  tarch::la::Vector<Dimensions, double> vertexX;
  tarch::la::Vector<Dimensions, double> prevVertexX;

  // Initialise values
  double vertexH_default = 3.;
  double dt = 1.;

  for (int i = 0; i < Dimensions; i++) {
    localPartX(i) = 51.5;
    dx(i) = -1.;
    vertexH(i) = vertexH_default;
  }

  int particleID = 1;
  int spacetreeId = 1;

  // Find initial pseudo-"vertex" particle is assigned to
  vertexX = internal::findVertexX(localPartX, vertexH);
  // Keep track of vertex index for identifiaction
  int vertexInd = internal::findVertexInd(localPartX(0), vertexH(0));

  // generate (minimal) particle
  ac::tests::TestParticle localPart =
      ac::tests::TestParticle(localPartX, particleID);
  // pretend to have a ParticleContainer
  std::vector<ac::tests::TestParticle *> assignedParticles = {&localPart};

  // First assignments
  ac::startMeshSweep("InitUnitTest");
  ac::assignParticleToVertex(
      particleName, localPartX, particleID,
      true, // this particle is always local
      vertexX, vertexH, spacetreeId, "initialAssign",
      false, // not a new particle
      true   // particle walks from vertex to vertex on same level
  );

  // Do some mesh sweeps.
  // NOTE: The database only keeps up to 16 sweeps in memory by default.
  // After that, entries will be purged. One sweep is added by default ('init').
  // We've already added another sweep ('InitUnitTest'). So if you go beyond 14,
  // it'll start deleting stuff before the end.
  size_t max_sweeps = 14;
  for (size_t sweep = 0; sweep < max_sweeps; sweep++) {

    ac::startMeshSweep("MySweep" + std::to_string(sweep));

    // record old particle positions.
    auto oldParticlePositions = ac::recordParticlePositions(assignedParticles);

    // insert tiny movement steps, make sure movement isn't recorded.
    double dt_use = dt;
    if (sweep > 8 and sweep < 12) {
      // TODO: fix this
      dt_use *= 1e-2;
    }

    // update particle position.
    localPartX += dt_use * dx;
    localPart.setX(localPartX);

    // Trace the movement.
    ac::traceParticleMovements(assignedParticles, oldParticlePositions, vertexX,
                               vertexH, spacetreeId);
    if (verbose)
      std::cout << ">>>>>>>>>>>> Moving particle to " << localPartX
                << std::endl;

    // pretend a particle can walk from vertex to vertex.
    // find current vertex particle is assigned to.
    int prevVertexInd = vertexInd;
    vertexInd = internal::findVertexInd(localPartX(0), vertexH(0));
    for (int i = 0; i < Dimensions; i++) {
      prevVertexX(i) = vertexX(i);
    }
    vertexX = internal::findVertexX(localPartX, vertexH);

    if (verbose)
      std::cout << ">>>>>>>>>>>> Checking vertex" << std::endl;

    if (vertexInd != prevVertexInd) {
      // Particle has changed vertex. Take note of that.
      if (verbose)
        std::cout << ">>>>>>>>>>>> Detaching vertex" << std::endl;
      ac::detachParticleFromVertex(
          particleName, localPart.getX(), localPart.getPartid(),
          true, // this particle is always local
          prevVertexX, vertexH, spacetreeId, "vertexChangeDetach");

      if (verbose)
        std::cout << ">>>>>>>>>>>> Attaching vertex" << std::endl;
      ac::assignParticleToVertex(
          particleName, localPart.getX(), localPart.getPartid(),
          true, // this particle is always local
          vertexX, vertexH, spacetreeId, "vertexChangeAssign",
          false, // not a new particle
          true   // particle walks from vertex to vertex on same level
      );
    }
  }

  if (verbose) {
    std::cout << ">>>>>>>>>>>> DATA DUMP END " << std::endl;
    std::cout << eventDatabase.toString() << std::endl;
  }

  // is particle count correct?
  assertion2(eventDatabase.getNumberOfTracedParticles() == 1,
             "Wrong particle count in database",
             eventDatabase.getNumberOfTracedParticles());

  // There is 1 more mesh sweep registered (intialization) than we
  // actually do.
  assertion3(eventDatabase.getCurrentMeshSweepIndex() == max_sweeps + 1,
             "Wrong mesh sweep count in database",
             eventDatabase.getCurrentMeshSweepIndex(), max_sweeps);

  ac::internal::ParticleSearchIdentifier identifier =
      ac::internal::ParticleSearchIdentifier("TestParticle", localPartX,
                                             particleID, vertexH(0));

  // 8 is hardcoded here. Modify manually if you change test setup.
  assertion3(eventDatabase.getTotalParticleEntries(identifier) == 8,
             "Wrong number of snapshots stored for particle",
             eventDatabase.getTotalParticleEntries(identifier),
             eventDatabase.particleHistory(identifier));

  // Clean up after yourself
  eventDatabase.reset();

#endif
}

void toolbox::particles::assignmentchecks::tests::TestHelpers::
    testParticleLiftDrop() {

#if PeanoDebug > 0
  const bool verbose = false;

  namespace ac = toolbox::particles::assignmentchecks;

  if (verbose)
    std::cout << ">>>>>>>>>>>> RUNNING testParticleLiftDrop() in " << Dimensions
              << "D" << std::endl;

  ac::internal::Database &eventDatabase = ac::internal::Database::getInstance();

  // IMPORTANT: Needs to be the same as the particle class name you use!
  // In traceParticleMovements(), this will be derived from the class name.
  std::string particleName = "TestParticle";

  // coordinates of particles
  tarch::la::Vector<Dimensions, double> localPartX;
  tarch::la::Vector<Dimensions, double> vertexH, prevVertexH;
  tarch::la::Vector<Dimensions, double> vertexX, prevVertexX;

  // Initialise values
  double vertexH_default = 3.;
  int depth = 3;
  int maxVertexDepth = 5;

  for (int i = 0; i < Dimensions; i++) {
    localPartX(i) = 51.5;
    vertexH(i) = vertexH_default;
    vertexX(i) = 27.;
  }

  int particleID = 1;
  int spacetreeId = 1;

  // generate (minimal) particle
  ac::tests::TestParticle localPart =
      ac::tests::TestParticle(localPartX, particleID);
  // pretend to have a ParticleContainer
  std::vector<ac::tests::TestParticle *> assignedParticles = {&localPart};

  // First assignments
  ac::startMeshSweep("InitUnitTest");
  ac::assignParticleToVertex(particleName, localPartX, particleID,
                             true, // this particle is always local
                             vertexX, vertexH, spacetreeId, "initialAssign");

  if (verbose)
    std::cout << ">>>>>>>>>>>>>>> Initially assigning to H=" << vertexH
              << " X=" << vertexX << " depth=" << depth << std::endl;

  ac::startMeshSweep("Lifts");

  // Does the particle need lifting from the current vertex?
  while (internal::liftParticle(localPartX, vertexX, vertexH, depth)) {
    depth--;
    prevVertexH = vertexH;
    prevVertexX = vertexX;
    vertexH *= 3.;
    vertexX = internal::findVertexX(vertexX, vertexH);

    if (verbose)
      std::cout << ">>>>>>>>>>>>>>> Lifting to H=" << vertexH
                << " X=" << vertexX << " depth=" << depth << std::endl;

    ac::detachParticleFromVertex(
        particleName, localPart.getX(), localPart.getPartid(),
        true, // this particle is always local
        prevVertexX, prevVertexH, spacetreeId, "vertexDetachLift");

    ac::assignParticleToVertex(
        particleName, localPart.getX(), localPart.getPartid(),
        true, // this particle is always local
        vertexX, vertexH, spacetreeId, "vertexAssignLift");
  }

  ac::startMeshSweep("Drops");

  // Now drop it in the correct vertex.
  while (internal::dropParticle(localPartX, vertexH, depth, maxVertexDepth)) {
    depth++;
    prevVertexH = vertexH;
    prevVertexX = vertexX;
    vertexH /= 3.;
    vertexX = internal::findVertexX(localPartX, vertexH);
    if (verbose)
      std::cout << ">>>>>>>>>>>>>>> Dropping to H=" << vertexH
                << " X=" << vertexX << " depth=" << depth << std::endl;

    ac::detachParticleFromVertex(
        particleName, localPart.getX(), localPart.getPartid(),
        true, // this particle is always local
        prevVertexX, prevVertexH, spacetreeId, "vertexDetachDrop");

    ac::assignParticleToVertex(
        particleName, localPart.getX(), localPart.getPartid(),
        true, // this particle is always local
        vertexX, vertexH, spacetreeId, "vertexAssignDrop");
  }

  // is particle count correct?
  assertion2(eventDatabase.getNumberOfTracedParticles() == 1,
             "Wrong particle count in database",
             eventDatabase.getNumberOfTracedParticles());

  ac::internal::ParticleSearchIdentifier identifier =
      ac::internal::ParticleSearchIdentifier("TestParticle", localPartX,
                                             particleID, vertexH(0));

  // 12 is hardcoded here. Modify manually if you change test setup.
  assertion3(eventDatabase.getTotalParticleEntries(identifier) == 13,
             "Wrong number of snapshots stored for particle",
             eventDatabase.getTotalParticleEntries(identifier),
             eventDatabase.particleHistory(identifier));

  if (verbose) {
    std::cout << ">>>>>>>>>>>> DATA DUMP END " << std::endl;
    std::cout << eventDatabase.toString() << std::endl;
  }
  // Clean up after yourself
  eventDatabase.reset();

#endif
}

void toolbox::particles::assignmentchecks::tests::TestHelpers::
    testParticleWalk() {

#if PeanoDebug > 0
  const bool verbose = true;

  namespace ac = toolbox::particles::assignmentchecks;

  if (verbose)
    std::cout << ">>>>>>>>>>>> RUNNING testLongParticleWalk() in " << Dimensions
              << "D" << std::endl;

  ac::internal::Database &eventDatabase = ac::internal::Database::getInstance();

  // IMPORTANT: Needs to be the same as the particle class name you use!
  // In traceParticleMovements(), this will be derived from the class name.
  std::string particleName = "TestParticle";

  // coordinates of particles
  tarch::la::Vector<Dimensions, double> localPartX;
  // particle displacement each step
  tarch::la::Vector<Dimensions, double> dx;
  tarch::la::Vector<Dimensions, double> vertexH, prevVertexH;
  tarch::la::Vector<Dimensions, double> vertexX, prevVertexX;

  // Initialise values
  double vertexH_default = 3.;
  double dt = 0.01;
  int depth = 4;
  int maxVertexDepth = 6;

  for (int i = 0; i < Dimensions; i++) {
    localPartX(i) = 51.5;
    dx(i) = -1.;
    vertexH(i) = vertexH_default;
  }

  int particleID = 1;
  int spacetreeId = 1;

  // Find initial pseudo-"vertex" particle is assigned to.
  vertexX = internal::findVertexX(localPartX, vertexH);

  // generate (minimal) particle
  ac::tests::TestParticle localPart =
      ac::tests::TestParticle(localPartX, particleID);
  // pretend to have a ParticleContainer
  std::vector<ac::tests::TestParticle *> assignedParticles = {&localPart};

  // First assignments
  ac::startMeshSweep("InitUnitTest");
  ac::assignParticleToVertex(particleName, localPartX, particleID,
                             true, // this particle is always local
                             vertexX, vertexH, spacetreeId, "initialAssign");

  // Do some mesh sweeps.
  // NOTE: The database only keeps up to 16 sweeps in memory by default.
  // After that, entries will be purged. One sweep is added by default ('init').
  // We've already added another sweep ('InitUnitTest'). So if you go beyond 14,
  // it'll start deleting stuff before the end.
  for (int sweep = 0; sweep < 1000; sweep++) {

    ac::startMeshSweep("MySweep" + std::to_string(sweep));

    // record old particle positions.
    auto oldParticlePositions = ac::recordParticlePositions(assignedParticles);

    // insert tiny movement steps, make sure movement isn't recorded.
    double dt_use = dt;
    if (sweep > 9 and sweep < 256) {
      dt_use *= 1e-2;
    }

    double dx_this_step = dt_use * dx(0);
    // Make sure we're not breaking our own rules.
    // assignmentchecks only work for small dx.
    assertion(dx_this_step <= vertexH(0));

    // update particle position.
    localPartX += dx_this_step;
    localPart.setX(localPartX);

    // Trace the movement.
    ac::traceParticleMovements(assignedParticles, oldParticlePositions, vertexX,
                               vertexH, spacetreeId);
    if (verbose)
      std::cout << ">>>>>>>>>>>> Moving particle to " << localPartX
                << std::endl;

    if (verbose)
      std::cout << ">>>>>>>>>>>> Checking vertex H=" << vertexH
                << " X=" << vertexX << std::endl;

    // Does the particle need lifting from the current vertex?
    while (internal::liftParticle(localPartX, vertexX, vertexH, depth)) {
      depth--;
      prevVertexH = vertexH;
      prevVertexX = vertexX;
      vertexH *= 3.;
      vertexX = internal::findVertexX(localPartX, vertexH);

      if (verbose)
        std::cout << ">>>>>>>>>>>>>>> Lifting to H=" << vertexH
                  << " X=" << vertexX << " depth=" << depth << std::endl;

      ac::detachParticleFromVertex(
          particleName, localPart.getX(), localPart.getPartid(),
          true, // this particle is always local
          prevVertexX, prevVertexH, spacetreeId, "vertexDetachLift");

      ac::assignParticleToVertex(
          particleName, localPart.getX(), localPart.getPartid(),
          true, // this particle is always local
          vertexX, vertexH, spacetreeId, "vertexAssignLift");
    }

    // Now drop it in the correct vertex.
    while (internal::dropParticle(localPartX, vertexH, depth, maxVertexDepth)) {
      depth++;
      prevVertexH = vertexH;
      prevVertexX = vertexX;
      vertexH /= 3.;
      vertexX = internal::findVertexX(localPartX, vertexH);
      if (verbose)
        std::cout << ">>>>>>>>>>>>>>> Dropping to H=" << vertexH
                  << " X=" << vertexX << " depth=" << depth << std::endl;

      ac::detachParticleFromVertex(
          particleName, localPart.getX(), localPart.getPartid(),
          true, // this particle is always local
          prevVertexX, prevVertexH, spacetreeId, "vertexDetachDrop");

      ac::assignParticleToVertex(
          particleName, localPart.getX(), localPart.getPartid(),
          true, // this particle is always local
          vertexX, vertexH, spacetreeId, "vertexAssignDrop");
    }
  }

  if (verbose) {
    std::cout << ">>>>>>>>>>>> DATA DUMP END " << std::endl;
    std::cout << eventDatabase.toString() << std::endl;
  }

  // Clean up after yourself
  eventDatabase.reset();

#endif
}

void toolbox::particles::assignmentchecks::tests::TestHelpers::
    testPeriodicBoundaryConditions() {

#if PeanoDebug > 0
  const bool verbose = false;

  namespace ac = toolbox::particles::assignmentchecks;

  if (verbose)
    std::cout << ">>>>>>>>>>>> RUNNING testPeriodicBoundaryConditions() in "
              << Dimensions << "D" << std::endl;

  ac::internal::Database &eventDatabase = ac::internal::Database::getInstance();
  eventDatabase.setMaxParticleSnapshotsToKeepTrackOf(1000);

  // IMPORTANT: Needs to be the same as the particle class name you use!
  // In traceParticleMovements(), this will be derived from the class name.
  std::string particleName = "TestParticle";

  // (Initial) particle position
  tarch::la::Vector<Dimensions, double> localPartX_init;
  // particle displacement each step
  tarch::la::Vector<Dimensions, double> dx;
  // position of boundaries.
  // This should be a plane, but in this simplified test,
  // where the particle is always travelling diagonally,
  // a single point does the trick.
  tarch::la::Vector<Dimensions, double> boundary;
  tarch::la::Vector<Dimensions, double> zero_corner;

  tarch::la::Vector<Dimensions, double> vertexH_init, prevVertexH;
  tarch::la::Vector<Dimensions, double> vertexX, prevVertexX;

  // Initialise values
  double vertexH_default = 3.;
  double dt = 1.;
  int depth = 4;
  int maxVertexDepth = 5;

  for (int i = 0; i < Dimensions; i++) {
    localPartX_init(i) = 7.5;
    dx(i) = -0.3;
    vertexH_init(i) = vertexH_default;
    boundary(i) = 100.;
    zero_corner(i) = 0.;
  }

  int particleID = 1;
  int spacetreeId = 1;

  // generate (minimal) particle
  ac::tests::TestParticle *localPart =
      new ac::tests::TestParticle(localPartX_init, particleID, true);
  localPart->setVertexH(vertexH_init);
  localPart->setDepth(depth);

  // pretend to have a ParticleContainer
  std::vector<ac::tests::TestParticle *> particleSet = {localPart};

  // Find initial pseudo-"vertex" particle is assigned to.
  vertexX = internal::findVertexX(localPart->getX(), localPart->getVertexH());

  // First assignments
  ac::startMeshSweep("InitUnitTest");
  ac::assignParticleToVertex(particleName, localPart->getX(),
                             localPart->getPartid(), localPart->isLocal(),
                             vertexX, localPart->getVertexH(), spacetreeId,
                             "initialAssign");

  // Do some mesh sweeps.
  // NOTE: The database only keeps up to 16 sweeps in memory by default.
  // After that, entries will be purged. One sweep is added by default ('init').
  // We've already added another sweep ('InitUnitTest'). So if you go beyond 14,
  // it'll start deleting stuff before the end.
  for (int sweep = 0; sweep < 50; sweep++) {

    ac::startMeshSweep("MySweep" + std::to_string(sweep));

    // insert tiny movement steps, make sure movement isn't recorded.
    double dt_use = dt;
    // if (sweep > 8 and sweep < 12) {
    //   dt_use *= ac::internal::ParticleIdentifier::Precision * 1e-2;
    // }

    // record old particle positions.
    auto oldParticlePositions = ac::recordParticlePositions(particleSet);

    // Now drift
    for (auto p : particleSet) {

      // Grab the vertex X before you move the particle! You need
      // to know what vertex it's currently assigned to, not what
      // it may be assigned to later.
      vertexX = internal::findVertexX(p->getX(), p->getVertexH());

      // update particle position.
      tarch::la::Vector<Dimensions, double> partX = p->getX();
      partX += dt_use * dx;
      p->setX(partX);

      // Trace the movement.
      ac::traceParticleMovements(particleSet, oldParticlePositions, vertexX,
                                 p->getVertexH(), spacetreeId);
      if (verbose)
        std::cout << ">>>>>>>>>>>> Moving particle to " << partX << std::endl;

      if (verbose)
        std::cout << ">>>>>>>>>>>> Checking vertex H=" << p->getVertexH()
                  << " X=" << vertexX << std::endl;

      // Does the particle need lifting from the current vertex?
      while (internal::liftParticle(p->getX(), vertexX, p->getVertexH(),
                                    p->getDepth())) {
        p->setDepth(p->getDepth() - 1);
        prevVertexH = p->getVertexH();
        prevVertexX = vertexX;
        p->setVertexH(p->getVertexH() * 3.);
        vertexX = internal::findVertexX(p->getX(), p->getVertexH());

        if (verbose)
          std::cout << ">>>>>>>>>>>>>>> Lifting to H=" << p->getVertexH()
                    << " X=" << vertexX << " depth=" << p->getDepth()
                    << std::endl;

        ac::detachParticleFromVertex(particleName, p->getX(), p->getPartid(),
                                     p->isLocal(), prevVertexX, prevVertexH,
                                     spacetreeId, "vertexDetach::Lift");

        ac::assignParticleToVertex(particleName, p->getX(), p->getPartid(),
                                   p->isLocal(), vertexX, p->getVertexH(),
                                   spacetreeId, "vertexAssign::Lift");
      }

      // Now drop it in the correct vertex.
      while (internal::dropParticle(p->getX(), p->getVertexH(), p->getDepth(),
                                    maxVertexDepth)) {
        p->setDepth(p->getDepth() + 1);
        prevVertexH = p->getVertexH();
        prevVertexX = vertexX;
        p->setVertexH(p->getVertexH() / 3.);
        vertexX = internal::findVertexX(p->getX(), p->getVertexH());

        if (verbose)
          std::cout << ">>>>>>>>>>>>>>> Dropping to H=" << p->getVertexH()
                    << " X=" << vertexX << " depth=" << p->getDepth()
                    << std::endl;

        ac::detachParticleFromVertex(particleName, p->getX(), p->getPartid(),
                                     p->isLocal(), prevVertexX, prevVertexH,
                                     spacetreeId, "vertexDetach::Drop");

        ac::assignParticleToVertex(particleName, p->getX(), p->getPartid(),
                                   p->isLocal(), vertexX, p->getVertexH(),
                                   spacetreeId, "vertexAssign::Drop");
      }
    } // Drift loop

    // Do we need to create a virtual particle?
    for (auto p : particleSet) {
      // particle always travels diagonally. Only check 1 coordinate.
      int vertexInd = internal::findVertexInd(p->getX()[0], p->getVertexH()[0]);
      if (vertexInd == 0) {
        // we're at the (0, 0, 0) boundary vertex. Replicate the particle
        // across the boundary.
        tarch::la::Vector<Dimensions, double> virtualPartX =
            p->getX() + boundary;
        TestParticle *replica = new TestParticle(virtualPartX, p->getPartid(),
                                                 false // virtual particle
        );
        replica->setVertexH(p->getVertexH());
        replica->setDepth(p->getDepth());
        particleSet.push_back(replica);

        if (verbose)
          std::cout << ">>>>>>>>>>>>>>>>>>>> Creating virtual particle at "
                    << virtualPartX << std::endl;

        // Take note of newly created particle.
        // Assume virtual particle is always at the same depth in the tree
        // as the local particle, so you may re-use vertexH (but not vertexX)
        vertexX = internal::findVertexX(replica->getX(), replica->getVertexH());

        ac::assignParticleToVertex(particleName, replica->getX(),
                                   replica->getPartid(), // copy particle ID
                                   false, // this particle is always virtual
                                   vertexX, replica->getVertexH(),
                                   spacetreeId, // Say we only have 1 spacetree
                                   "createdVirtualParticle");
      }
    }

    // Better safe than sorry. We should only have 1 extra virtual particle at
    // the most.
    assertion(particleSet.size() <= 2);

    // Check ParallelState of particles:
    // Change local->virtual if crossed (0, 0, 0) boundary
    // change virtual->local if crossed boundary
    for (auto p : particleSet) {
      if (p->isLocal()) {
        if (tarch::la::oneSmaller(p->getX(), zero_corner)) {

          vertexX = internal::findVertexX(p->getX(), p->getVertexH());

          ac::detachParticleFromVertex(particleName, p->getX(), p->getPartid(),
                                       true, // here it's local
                                       vertexX, p->getVertexH(), spacetreeId,
                                       "vertexDetach::LocalToVirtual");

          ac::assignParticleToVertex(particleName, p->getX(), p->getPartid(),
                                     false, // now make it virtual
                                     vertexX, p->getVertexH(), spacetreeId,
                                     "vertexAssign::LocalToVirtual");

          p->setIsLocal(false);
        }
      } else {
        if (tarch::la::oneSmaller(p->getX(), boundary)) {
          // We're keeping the virtual particle.

          vertexX = internal::findVertexX(p->getX(), p->getVertexH());
          ac::detachParticleFromVertex(particleName, p->getX(), p->getPartid(),
                                       false, // here it's virtual
                                       vertexX, p->getVertexH(), spacetreeId,
                                       "vertexDetach::VirtualToLocal");

          ac::assignParticleToVertex(particleName, p->getX(), p->getPartid(),
                                     true, // now make it virtual
                                     vertexX, p->getVertexH(), spacetreeId,
                                     "vertexAssign::VirtualToLocal");

          p->setIsLocal(true);
        }
      }
    }

    // Delete the virtual particle.
    auto p = particleSet.begin();
    while (p != particleSet.end()) {
      if (not(*p)->isLocal()) {
        std::cout << " <<<<<<<<<<<<<<<<<<<<<<< ERASING " << (*p)->getX()
                  << std::endl;

        vertexX = internal::findVertexX((*p)->getX(), (*p)->getVertexH());

        ac::detachParticleFromVertex(
            particleName, (*p)->getX(), (*p)->getPartid(), (*p)->isLocal(),
            vertexX, (*p)->getVertexH(), spacetreeId, "vertexDetachErase");
        ac::eraseParticle(particleName, (*p)->getX(), (*p)->getPartid(),
                          (*p)->isLocal(), (*p)->getVertexH(), spacetreeId,
                          "eraseVirtualParticle");
        delete *p;
        p = particleSet.erase(p);
      } else {
        p++;
      }
    }

  } // mesh sweep loops

  // if (verbose) {
  std::cout << ">>>>>>>>>>>> DATA DUMP END " << std::endl;
  std::cout << eventDatabase.toString() << std::endl;
  // }

  // Clean up after yourself
  eventDatabase.reset();

#endif
}

void toolbox::particles::assignmentchecks::tests::TestHelpers::testSieveSet() {

#if PeanoDebug > 0
  const bool verbose = false;

  namespace ac = toolbox::particles::assignmentchecks;

  if (verbose)
    std::cout << ">>>>>>>>>>>> RUNNING testSieveSet() in " << Dimensions << "D"
              << std::endl;

  ac::internal::Database &eventDatabase = ac::internal::Database::getInstance();
  eventDatabase.setMaxParticleSnapshotsToKeepTrackOf(1000);

  // IMPORTANT: Needs to be the same as the particle class name you use!
  // In traceParticleMovements(), this will be derived from the class name.
  std::string particleName = "TestParticle";

  // (Initial) particle position
  tarch::la::Vector<Dimensions, double> localPartX_init;
  // particle displacement each step
  tarch::la::Vector<Dimensions, double> dx;

  tarch::la::Vector<Dimensions, double> vertexH, prevVertexH;
  tarch::la::Vector<Dimensions, double> vertexX, prevVertexX;

  // Initialise values
  double vertexH_default = 3.;
  double dt = 1.;

  for (int i = 0; i < Dimensions; i++) {
    localPartX_init(i) = 7.5;
    dx(i) = 2. * vertexH_default;
    vertexH(i) = vertexH_default;
  }

  int particleID = 1;
  int spacetreeId = 1;

  // generate (minimal) particle
  ac::tests::TestParticle *localPart =
      new ac::tests::TestParticle(localPartX_init, particleID, true);
  localPart->setVertexH(vertexH);
  localPart->setDepth(1); // dummy val

  // pretend to have a ParticleContainer
  std::vector<ac::tests::TestParticle *> particleSet = {localPart};

  // Find initial pseudo-"vertex" particle is assigned to.
  vertexX = internal::findVertexX(localPart->getX(), localPart->getVertexH());

  // First assignments
  ac::startMeshSweep("InitUnitTest");
  ac::assignParticleToVertex(particleName, localPart->getX(),
                             localPart->getPartid(), localPart->isLocal(),
                             vertexX, localPart->getVertexH(), spacetreeId,
                             "initialAssign");

  // Do some mesh sweeps.
  size_t nsweeps = 10;
  for (size_t sweep = 0; sweep < nsweeps; sweep++) {

    ac::startMeshSweep("MySweep" + std::to_string(sweep));

    // record old particle positions.
    auto oldParticlePositions = ac::recordParticlePositions(particleSet);

    // Now drift
    for (auto p : particleSet) {

      // Grab the vertex X before you move the particle! You need
      // to know what vertex it's currently assigned to, not what
      // it may be assigned to later.
      vertexX = internal::findVertexX(p->getX(), p->getVertexH());

      // update particle position.
      tarch::la::Vector<Dimensions, double> partX = p->getX();
      partX += dt * dx;
      p->setX(partX);

      // Trace the movement.
      ac::traceParticleMovements(particleSet, oldParticlePositions, vertexX,
                                 p->getVertexH(), spacetreeId);
      if (verbose)
        std::cout << ">>>>>>>>>>>> Moving particle to " << partX << std::endl;

      // First we detach the moved particle.
      prevVertexH = p->getVertexH();
      prevVertexX = vertexX;

      ac::detachParticleFromVertex(particleName, p->getX(), p->getPartid(),
                                   p->isLocal(), prevVertexX, prevVertexH,
                                   spacetreeId, "vertexDetach");

      // Assign it to the sieve set
      ac::assignParticleToSieveSet(particleName, p->getX(), p->getPartid(),
                                   p->isLocal(), p->getVertexH(), spacetreeId,
                                   "assignToSieveSet");
      // Now "sieve it" into the right vertex
      vertexX = internal::findVertexX(p->getX(), p->getVertexH());

      ac::assignParticleToVertex(particleName, p->getX(), p->getPartid(),
                                 p->isLocal(), vertexX, p->getVertexH(),
                                 spacetreeId, "vertexAssign::Sieve");
    } // Drift loop

  } // mesh sweep loops

  // is particle count correct?
  assertion2(eventDatabase.getNumberOfTracedParticles() == 1,
             "Wrong particle count in database",
             eventDatabase.getNumberOfTracedParticles());

  ac::internal::ParticleSearchIdentifier identifier =
      ac::internal::ParticleSearchIdentifier("TestParticle", localPart->getX(),
                                             localPart->getPartid(),
                                             vertexH(0));

  size_t nEntries = eventDatabase.getTotalParticleEntries(identifier);
  // we add 4 events per sweep, and 1 initial one
  assertion5(nEntries == (nsweeps * 4 + 1),
             "Wrong number of entries for particle ", nEntries, nsweeps * 4 + 1,
             eventDatabase.particleHistory(identifier),
             eventDatabase.toString());

  if (verbose) {
    // Print out particle histories
    std::cout << eventDatabase.particleHistory(identifier);
  }

  // if (verbose) {
  std::cout << ">>>>>>>>>>>> DATA DUMP END " << std::endl;
  std::cout << eventDatabase.toString() << std::endl;
  // }

  // Clean up after yourself
  eventDatabase.reset();

#endif
}

// void toolbox::particles::assignmentchecks::tests::TestHelpers::run() {
// TODO: add everything from main.cpp in here
// testMethod(testTruthTableSearchAndIDKeys(verbose));
// testMethod(testAddingSweepsToDatabase(verbose));
// testMethod(testAddingParticleEvents(verbose));
// runner.testAddingParticleMovingEvents(100, 1000, verbose);
// runner.testAddingParticleMovingEvents(100, 16, verbose);
// testMethod(testParticleWalkSameTreeLevel);
// testMethod(testParticleLiftDrop);
// testMethod(testParticleWalk);
// testMethod(testLongParticleWalk);
// testMethod(testPeriodicBoundaryConditions);
// testMethod(testSieveSet);
// }

#ifdef UseTestSpecificCompilerSettings
#pragma optimize("", on)
#endif
