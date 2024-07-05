#include "Database.h"
// #include "ParticleIdentifier.h"
#include "Event.h"
#include "VectorVectorOperations.h"

#include <iterator>
#include <limits>
#include <ranges>
#include <vector>

// #include "tarch/multicore/MultiReadSingleWriteLock.h"


#if !defined(AssignmentChecks) and !defined(noAssignmentChecks) \
  and PeanoDebug > 0
#define AssignmentChecks
#endif

namespace {
// TODO: Add this back
  // tarch::logging::Log _log("toolbox::particles::assignmentchecks");

  toolbox::particles::assignmentchecks::internal::Database _database;
} // namespace



// checked
toolbox::particles::assignmentchecks::internal::Database::Database(
  size_t maxParticleSnapshotsToKeepTrackOf
):
  _maxParticleSnapshotsToKeepTrackOf(maxParticleSnapshotsToKeepTrackOf) {
    _currentMeshSweepIndex = 0;
    _meshSweepData.push_back(MeshSweepData("initial"));
};


// checked
void toolbox::particles::assignmentchecks::internal::Database::startMeshSweep(
  const std::string& meshSweepName
) {

  // TODO: Re-Insert
  // logInfo(
  //   "startMeshSweep()",
  //   "finish old mesh sweep with "
  //     << _data.rbegin()->size() << " event(s) and start new one for "
  //     << meshSweepName
  // );
  // tarch::multicore::MultiReadSingleWriteLock
  //   lock(_semaphore, tarch::multicore::MultiReadSingleWriteLock::Write);
  _meshSweepData.push_back(MeshSweepData(meshSweepName));
  _currentMeshSweepIndex++;
};


// checked
int toolbox::particles::assignmentchecks::internal::Database::
  getNumberOfSnapshots() const {
  return _data.size();
}


// checked
std::vector<toolbox::particles::assignmentchecks::internal::MeshSweepData>& toolbox::particles::assignmentchecks::internal::Database::
  getMeshSweepData() {
  return _meshSweepData;
}


// checked
size_t toolbox::particles::assignmentchecks::internal::Database::getCurrentMeshSweepIndex() const {
  return _currentMeshSweepIndex;
}


// checked
int toolbox::particles::assignmentchecks::internal::Database::getNumberOfTracedParticles() const {
  return _data.size();
}


// checked
void toolbox::particles::assignmentchecks::internal::Database::
  reset() {

  _data.clear();
  _meshSweepData.clear();

  // and re-initialize.
  _meshSweepData.push_back(MeshSweepData("initial"));
  _currentMeshSweepIndex = 0;
}


toolbox::particles::assignmentchecks::internal::ParticleEvents&
  toolbox::particles::assignmentchecks::internal::Database::getParticleHistory(
    const ParticleSearchIdentifier& identifier
  ) {
  // tarch::multicore::MultiReadSingleWriteLock
  //   lock(_semaphore, tarch::multicore::MultiReadSingleWriteLock::Read);

  auto search = _data.find(identifier);

  if (search == _data.end()){
    ParticleEvents *newEventHistory = new ParticleEvents();
    newEventHistory->push_back(Event(Event::Type::NotFound));
    return *newEventHistory;
  }
  else {

    ParticleEvents& history = search->second;
    return history;
  }
}



// checked
std::string toolbox::particles::assignmentchecks::internal::Database::toString(
) {
  // TODO: Add back in
  // tarch::multicore::MultiReadSingleWriteLock
  //   lock(_semaphore, tarch::multicore::MultiReadSingleWriteLock::Read);

  std::ostringstream msg;

  msg << "--------------------------\n";
  msg << "Full database dump\n";
  msg << "--------------------------\n";

  for (auto entry = _data.cbegin(); entry != _data.cend(); entry++){
    ParticleIdentifier identifier = entry->first;
    ParticleEvents history = entry->second;

    msg << "\n" << identifier.toString();
    int meshSweepIndex = -1;

    for (auto event = history.crbegin(); event != history.crend(); event++){
      if (event->meshSweepIndex != meshSweepIndex){
        std::string sweepname = _meshSweepData.at(event->meshSweepIndex).getName();
        msg << "\n\t[Sweep " << sweepname << "]:";
      }
      meshSweepIndex = event->meshSweepIndex;
      msg << "\n\t\t->" << event->toString();
    }
  }

  msg << "\n";

  return msg.str();
}


// checked
int toolbox::particles::assignmentchecks::internal::Database::getTotalParticleEntries(
  const ParticleSearchIdentifier& identifier
) {
  // tarch::multicore::MultiReadSingleWriteLock
  //   lock(_semaphore, tarch::multicore::MultiReadSingleWriteLock::Read);

  auto search = _data.find(identifier);
  if (search != _data.end()){
    int result = search->second.size();
    return result;
  }
  return 0;
}


std::string toolbox::particles::assignmentchecks::internal::Database::
  lastMeshSweepSnapshot() {
  // tarch::multicore::MultiReadSingleWriteLock
  //   lock(_semaphore, tarch::multicore::MultiReadSingleWriteLock::Read);
  assert(false);

  std::ostringstream msg;
    // if (not _data.empty()) {
    /* const auto& lastMeshSnapshot = *_data.crbegin(); */
    /* msg */
    /*   << "#" << (_data.size() - 1) << "(" << lastMeshSnapshot.getName() << "):"; */
    /* for (const auto& particleTrace : lastMeshSnapshot) { */
    /*   msg << std::endl << "-" << particleTrace.first.toString() << ": "; */
    /*   for (const auto& event : particleTrace.second) { */
    /*     msg << event.toString(); */
    /*   } */
    /* } */
  /* } */
  return msg.str();
}


// checked
std::string toolbox::particles::assignmentchecks::internal::Database::sweepHistory() const {

  std::ostringstream msg;
  int counter = 0;
  for (auto sweep = _meshSweepData.cbegin(); sweep != _meshSweepData.cend(); sweep++){
    msg << std::endl << "sweep #" << counter << ": " << sweep->getName() ;
    counter++;
  }
  msg << "\n";

  return msg.str();
}


// checked
std::string toolbox::particles::assignmentchecks::internal::Database::
  particleHistory(const ParticleSearchIdentifier& identifier) {
  // tarch::multicore::MultiReadSingleWriteLock
  //   lock(_semaphore, tarch::multicore::MultiReadSingleWriteLock::Read);

  std::ostringstream msg;
  msg
    << "\n============================\n"
    << identifier.toString()
    << "\n============================\n";

  auto search = _data.find(identifier);
  if (search != _data.end()){

    ParticleEvents& history = search->second;
    auto ev = history.crbegin();

    // mesh sweep index of the previous event.
    int prevMeshSweepInd = -1;

    while (ev != history.crend()){

      int meshSweepInd = ev->meshSweepIndex;

      if (meshSweepInd != prevMeshSweepInd){
        // We're starting new sweep. Print header.
        msg << "\nsweep #" << meshSweepInd <<
          " (" << _meshSweepData.at(meshSweepInd).getName() << "):";
      }
      msg << "\n\t->" << ev->toString();

      prevMeshSweepInd = meshSweepInd;
      ev++;
    }
  }

  msg << "\n";

  return msg.str();
}

// checked
toolbox::particles::assignmentchecks::internal::ParticleEvents& toolbox::particles::assignmentchecks::internal::Database::addEvent(
  ParticleSearchIdentifier identifier,
  Event&       event
) {
  // tarch::multicore::MultiReadSingleWriteLock
  //   lock(_semaphore, tarch::multicore::MultiReadSingleWriteLock::Write);
  //

  ParticleEvents output;

  // Take note of the current mesh sweep index.
  event.meshSweepIndex = _database.getCurrentMeshSweepIndex();

  auto search = _data.find(identifier);

  if (search == _data.end()){
    // This is a new particle.

    ParticleEvents newEventHistory = ParticleEvents();
    newEventHistory.push_back(event);
    _data.insert(std::pair<ParticleIdentifier, ParticleEvents>( identifier, newEventHistory ) );
    // TODO: Re-Insert
    // logDebug(
std::cout <<
      "addEvent(...) " <<
      "add new particle history thread for "
        << identifier.toString()
<< std::endl;
    // );
    return _data.at(identifier);

  } else {

    ParticleEvents& history = search->second;

    // First: Do we need to downsize the history?
    if (history.size() >= _maxParticleSnapshotsToKeepTrackOf){

      Event last = history.back();

      // TODO: Re-Insert
      // logDebug(
std::cout <<
        "addEvent(...) " <<
        "shortening history to " << last.toString()
<< std::endl;
      // );

      last.trace = "substitute-for-whole-trajectory/" + last.trace;
      history.clear();
      history.push_back(last);
    }

    // Add the new event now.
    history.push_back(event);
std::cout <<
        "addEvent(...) " <<
        "added event " << event.toString() << identifier.particleX
<< std::endl;


    // Do we need to shift the coordinates of the identifier?
    ParticleIdentifier key = search->first;
    // TODO MLADEN: use tarch::la::oneGreater here
    bool shift = false;
    for (int i = 0; i < Dimensions; i++){
      if (std::abs(key.particleX(i) - identifier.particleX(i)) > ParticleSearchIdentifier::shiftTolerance * identifier.positionTolerance){
        shift = true;
      }
    }

    if (shift){
      // delete old entry in database and add the new one.
      // TODO: Re-Insert
      // logDebug(
std::cout <<
        "addEvent(...) " <<
        "shifting particle identifier from "
          << key.particleX << " to " << identifier.particleX
<< std::endl;
      // );

      ParticleEvents historyCopy = history;
      _data.erase(key);
      _data.insert(std::pair<ParticleIdentifier, ParticleEvents>( identifier, historyCopy));
    }

    return history;
  }
}

// checked
toolbox::particles::assignmentchecks::internal::Event toolbox::particles::assignmentchecks::internal::getPreviousEvent(ParticleEvents& particleHistory, int spacetreeId, size_t nFirstEventsToSkip) {

  if (particleHistory.size() <= nFirstEventsToSkip){
    return Event(Event::Type::NotFound);
  }

  auto it = particleHistory.crbegin();
  std::advance(it, nFirstEventsToSkip);

  if (it == particleHistory.crend()){
    return Event(Event::Type::NotFound);
  }

  if (spacetreeId == Database::AnyTree) return *it;

  while (it != particleHistory.crend()){
    if (it->treeId == spacetreeId) return *it;
    it++;
  }

  return Event(Event::Type::NotFound);
}

#if defined(AssignmentChecks)

// checked
std::string toolbox::particles::assignmentchecks::sweepHistory() {
  return _database.sweepHistory();
}

// checked
void toolbox::particles::assignmentchecks::startMeshSweep(
  const std::string& meshSweepName
) {
  _database.startMeshSweep(meshSweepName);
}

// checked
void toolbox::particles::assignmentchecks::eraseParticle(
  const std::string&                           particleName,
  const tarch::la::Vector<Dimensions, double>& particleX,
  const int                                    particleID,
  bool                                         isLocal,
  const tarch::la::Vector<Dimensions, double>& vertexH,
  int                                          treeId,
  const std::string&                           trace
) {

  // TODO: Re-Insert
  // logTraceInWith4Arguments(
  //   "eraseParticle(...)",
  //   particleName,
  //   particleX,
  //   isLocal,
  //   treeId
  // );

  using namespace internal;

  ParticleSearchIdentifier identifier = ParticleSearchIdentifier(
      particleName,
      particleX,
      particleID,
      tarch::la::max(vertexH)
  );

  Event event(
    Event::Type::Erase,
    isLocal,
    treeId,
    trace
  );

  ParticleEvents& history = _database.addEvent(identifier, event);
  Event previousEvent = getPreviousEvent(history, treeId, 1);


  assertion5(
    previousEvent.type == internal::Event::Type::DetachFromVertex,
    identifier.toString(),
    event.toString(),
    previousEvent.toString(),
    treeId,
    _database.particleHistory(identifier)
  );

  // TODO: Re-Insert
  // logTraceOut("eraseParticle(...)");
}


// checked
void toolbox::particles::assignmentchecks::assignParticleToVertex(
  const std::string&                           particleName,
  const tarch::la::Vector<Dimensions, double>& particleX,
  const int                                    particleID,

  bool                                         isLocal,
  const tarch::la::Vector<Dimensions, double>& vertexX,
  const tarch::la::Vector<Dimensions, double>& vertexH,
  int                                          treeId,
  const std::string&                           trace,
  bool                                         particleIsNew,
  bool                                         reassignmentOnSameTreeDepthAllowed
) {

    // TODO: Re-Insert
  // logTraceInWith6Arguments(
  //   "assignParticleToVertex(...)",
  //   particleName,
  //   particleX,
  //   isLocal,
  //   vertexX,
  //   vertexH,
  //   treeId
  // );

  using namespace internal;

  constexpr bool checkNewParticles = false;

  ParticleSearchIdentifier identifier = ParticleSearchIdentifier(
      particleName,
      particleX,
      particleID,
      tarch::la::max(vertexH)
  );

  Event event(
    Event::Type::AssignToVertex,
    isLocal,
    vertexX,
    particleX,
    vertexH,
    treeId,
    trace
  );

  ParticleEvents& history = _database.addEvent(identifier, event);

  if (not (particleIsNew and (not checkNewParticles))){
    // skip the newly added event.
    Event previousEvent = getPreviousEvent(history, treeId, 1);

    const bool isDropping = previousEvent.type == internal::Event::Type::DetachFromVertex
                            and tarch::la::allGreater(previousEvent.vertexH, vertexH);
    const bool isLifting = previousEvent.type == internal::Event::Type::DetachFromVertex
                           and tarch::la::allSmaller(previousEvent.vertexH, vertexH);
    const bool isDroppingFromSieveSet = previousEvent.type == internal::Event::Type::AssignToSieveSet;

    const bool isDetached = reassignmentOnSameTreeDepthAllowed ?
        (previousEvent.type == internal::Event::Type::DetachFromVertex) :
        (previousEvent.type == internal::Event::Type::DetachFromVertex and not previousEvent.isLocal);

    const bool virtualPartThatHasBeenErased =
        (previousEvent.type == internal::Event::Type::Erase and (not previousEvent.isLocal));

    if (isLocal) {
      assertion7(
        previousEvent.type == internal::Event::Type::NotFound
        or
        isDroppingFromSieveSet
        or
        (isLifting and previousEvent.isLocal)
        or
        (isDropping and previousEvent.isLocal)
        or
        (isDetached),
        identifier.toString(),
        event.toString(),
        previousEvent.toString(),
        treeId,
        _database.getNumberOfSnapshots(),
        trace,
        _database.particleHistory(identifier)
      );
    } else {
        assertion7(
          (previousEvent.type == internal::Event::Type::NotFound)
          or
          isDroppingFromSieveSet
          or
          (isDropping and not previousEvent.isLocal)
          or
          (previousEvent.type == internal::Event::Type::DetachFromVertex and previousEvent.isLocal)
          or virtualPartThatHasBeenErased,
          identifier.toString(),
          event.toString(),
          previousEvent.toString(),
          treeId,
          _database.getNumberOfSnapshots(),
          trace,
          _database.particleHistory(identifier)
        );
    }
  }

  // TODO: Re-Insert
  // logTraceOut("assignParticleToVertex(...)");
}

// checked
void toolbox::particles::assignmentchecks::moveParticle(
  const std::string&                           particleName,
  const tarch::la::Vector<Dimensions, double>& oldParticleX,
  const tarch::la::Vector<Dimensions, double>& newParticleX,
  const int                                    particleID,
  const tarch::la::Vector<Dimensions, double>& vertexX,
  const tarch::la::Vector<Dimensions, double>& vertexH,
  int                                          treeId,
  const std::string&                           trace
) {

  // TODO: Re-Insert
  // logTraceInWith3Arguments(
  //   "moveParticle(...)",
  //   particleName,
  //   oldParticleX,
  //   newParticleX
  // );

  using namespace internal;

  // use old particle position to find history.
  ParticleSearchIdentifier identifier = ParticleSearchIdentifier(
      particleName,
      newParticleX,
      particleID,
      tarch::la::max(vertexH)
  );

  ParticleEvents& history = _database.getParticleHistory(identifier);
  // We assume that we can't be moving a particle without having done anything else first.
  assert(history.size() > 0 );

  Event previousEvent = getPreviousEvent(history, treeId);
  Event anyPreviousEvent = getPreviousEvent(history, Database::AnyTree);

  // We need to have at least 1 event somewhere.
  // Even if the database had been trimmed, at least 1 event remains. But the remaining
  // event may have been on a different tree.
  assertion(anyPreviousEvent.type != internal::Event::Type::NotFound);

  // We must still be on the same vertex, or something is wrong.
  assertion3(anyPreviousEvent.vertexH == vertexH,
      "Vertices not the same",
      anyPreviousEvent.vertexH,
      vertexH
      );
  assertion3(anyPreviousEvent.vertexX == vertexX,
      "Vertices not the same",
      anyPreviousEvent.vertexX,
      vertexX
      );

  if (previousEvent.type != Event::Type::NotFound){
    // This must be on the same tree then.
    // We can apply stricter checks.

    // TODO: Check sieve set sieve set assignment
    assertion(
        previousEvent.type == Event::Type::AssignToVertex or
        previousEvent.type == Event::Type::MoveWhileAssociatedToVertex or
        previousEvent.type == Event::Type::ConsecutiveMoveWhileAssociatedToVertex
    );

    // We must still be on the same vertex, or something is wrong.
    assertion3(previousEvent.vertexH == vertexH,
        "Vertices not the same",
        previousEvent.vertexH,
        vertexH
        );
    assertion3(previousEvent.vertexX == vertexX,
        "Vertices not the same",
        previousEvent.vertexX,
        vertexX
        );
    // TODO: this should be redundant, Remove later.
    assertion3(previousEvent.treeId == treeId,
        "Tree not the same",
        previousEvent.treeId,
        treeId
        );
  }

  if ((previousEvent.type == Event::Type::MoveWhileAssociatedToVertex) or (previousEvent.type == Event::Type::ConsecutiveMoveWhileAssociatedToVertex)){

    Event newEvent = Event(Event::Type::NotFound);

    if (previousEvent.type == Event::Type::MoveWhileAssociatedToVertex) {
      // we turn it into a ConsecutiveMoveWhileAssociatedToVertex and
      // store the current particle position.
      std::ostringstream pastTrace;
      pastTrace << "Consecutive Move starting at x=[" <<
        // TODO: Add this back in
        // previousEvent.previousParticleX.toString() << "] at sweep " <<
        _database.getMeshSweepData().at(previousEvent.meshSweepIndex).getName() <<
        " | old trace: " << previousEvent.trace;
        " | new trace: " + trace;

      // Create new event.
      newEvent = Event(
          Event::Type::ConsecutiveMoveWhileAssociatedToVertex,
          previousEvent.isLocal,
          vertexX,
          oldParticleX,
          vertexH,
          treeId,
          pastTrace.str(),
          -1 // will be modified in _database.addEvent
          );
    } else {
      newEvent = Event(
          Event::Type::ConsecutiveMoveWhileAssociatedToVertex,
          previousEvent.isLocal,
          vertexX,
          previousEvent.previousParticleX,
          vertexH,
          treeId,
          previousEvent.trace,
          -1 // will be modified in _database.addEvent
          );
    }

    // In either case, delete old event from history and then re-add it so that
    // all the mechanisms in addEvent (shortening history, modifying
    // identifier coordinates) trigger.
    for (auto it = history.begin(); it != history.end(); it++){
      if ((it->meshSweepIndex == previousEvent.meshSweepIndex)
        and (it->type == previousEvent.type)
          and (it->treeId == previousEvent.treeId)
          and (it->isLocal == previousEvent.isLocal)){
        history.erase(it);
        break;
      }
    }

    // Add new event now.
    _database.addEvent(identifier, newEvent);
  }
  else {
    // Add a new move event.
    Event newEvent = Event(
        Event::Type::MoveWhileAssociatedToVertex,
        vertexX,
        oldParticleX,
        vertexH,
        treeId,
        trace
        );

    _database.addEvent(identifier, newEvent);
  }

  // TODO: Re-Insert
  // logTraceOut("moveParticle(...)");
}


void toolbox::particles::assignmentchecks::detachParticleFromVertex(
  const std::string&                           particleName,
  const tarch::la::Vector<Dimensions, double>& particleX,
  const int                                    particleID,
  bool                                         isLocal,
  const tarch::la::Vector<Dimensions, double>& vertexX,
  const tarch::la::Vector<Dimensions, double>& vertexH,
  int                                          treeId,
  const std::string&                           trace
) {

  // TODO: Re-Insert
  /* logTraceInWith6Arguments( */
    /* "detachParticleFromVertex(...)", */
    /* particleName, */
    /* particleX, */
    /* isLocal, */
    /* vertexX, */
    /* vertexH, */
    /* treeId */
  // );

  using namespace internal;

  ParticleSearchIdentifier identifier = ParticleSearchIdentifier(
      particleName,
      particleX,
      particleID,
      tarch::la::max(vertexH)
  );

  Event event(
    Event::Type::DetachFromVertex,
    isLocal,
    vertexX,
    particleX,
    vertexH,
    treeId,
    trace
  );

  ParticleEvents& history = _database.addEvent(identifier, event);
  Event previousEvent = getPreviousEvent(history, treeId, 1);


  assertion6(
    previousEvent.type == internal::Event::Type::AssignToVertex
    or previousEvent.type == internal::Event::Type::MoveWhileAssociatedToVertex
    or previousEvent.type == internal::Event::Type::ConsecutiveMoveWhileAssociatedToVertex,
    identifier.toString(),
    event.toString(),
    previousEvent.toString(),
    treeId,
    _database.getNumberOfSnapshots(),
    trace
  );
  assertion7(
    tarch::la::equals(previousEvent.vertexX, vertexX),
    identifier.toString(),
    event.toString(),
    previousEvent.toString(),
    treeId,
    _database.getNumberOfSnapshots(),
    trace,
    _database.particleHistory(identifier)
  );
  assertion6(
    tarch::la::equals(previousEvent.vertexH, vertexH),
    identifier.toString(),
    event.toString(),
    previousEvent.toString(),
    treeId,
    trace,
    _database.particleHistory(identifier)
  );

  // TODO: Re-Insert
  // logTraceOut("detachParticleFromVertex(...)");
}


void toolbox::particles::assignmentchecks::assignParticleToSieveSet(
  const std::string&                           particleName,
  const tarch::la::Vector<Dimensions, double>& particleX,
  const int                                    particleID,
  bool                                         isLocal,
  const tarch::la::Vector<Dimensions, double>& vertexH,
  int                                          treeId,
  const std::string&                           trace
) {

  assert(false);
  // TODO: Re-Insert
  /*   logTraceInWith4Arguments( */
  /*   "assignParticleToSieveSet(...)", */
  /*   particleName, */
  /*   particleX, */
  /*   isLocal, */
  /*   treeId */
  /* ); */
  /*  */
  /* logDebug( */
  /*   "assignParticleToSieveSet()", */
  /*   "assign " */
  /*     << particleName << " particle at " << particleX */
  /*     << " to global sieve set on tree " << treeId */
  /* ); */
  /*  */
  // internal::ParticleSearchIdentifier identifier = _database.createParticleSearchIdentifier(
  //   particleName,
  //   particleX,
  //   particleID,
  //   tarch::la::max(vertexH)
  // );
  // internal::Event
  //   event{internal::Event::Type::AssignToSieveSet, isLocal, treeId, trace};
  //
  // internal::Event previousEvent = _database.getEntry(identifier, treeId).first;
  //
  // assertion5(
  //   previousEvent.type == internal::Event::Type::DetachFromVertex,
  //   identifier.toString(),
  //   event.toString(),
  //   previousEvent.toString(),
  //   treeId,
  //   _database.particleHistory(identifier)
  // );
  //
  // _database.addEvent(identifier, event);
  // // TODO: Re-Insert
  // logTraceOut("assignParticleToSieveSet(...)");
}


// checked
void toolbox::particles::assignmentchecks::ensureDatabaseIsEmpty() {
  if (_database.getNumberOfSnapshots() != 0) {
  // TODO: Re-Insert
    // logInfo(
    //   "ensureDatabaseIsEmpty()"
    std::cout <<
      "database still holds " << _database.getNumberOfSnapshots()
      << " snapshots"
      << std::endl; // TODO: rm endl
    // );
    // logError("ensureDatabaseIsEmpty()", _database.toString());
    assertion(false);
    exit(-1);
  }
}

toolbox::particles::assignmentchecks::internal::Database& toolbox::particles::assignmentchecks::getDatabaseInstance(){
  return _database;
}

#else

void toolbox::particles::assignmentchecks::startMeshSweep(
  const std::string& meshSweepName
) {}

void toolbox::particles::assignmentchecks::eraseParticle(
  const std::string&                           particleName,
  const tarch::la::Vector<Dimensions, double>& particleX,
  const int                                    particleID,
  bool                                         isLocal,
  const tarch::la::Vector<Dimensions, double>& vertexH,
  int                                          treeId,
  const std::string&                           trace
) {}

void toolbox::particles::assignmentchecks::assignParticleToVertex(
  const std::string&                           particleName,
  const tarch::la::Vector<Dimensions, double>& particleX,
  const int                                    particleID,
  bool                                         isLocal,
  const tarch::la::Vector<Dimensions, double>& vertexX,
  const tarch::la::Vector<Dimensions, double>& vertexH,
  int                                          treeId,
  const std::string&                           trace,
  bool                                         particleIsNew,
  bool                                         reassignmentOnSameTreeDepthAllowed
) {}

void toolbox::particles::assignmentchecks::detachParticleFromVertex(
  const std::string&                           particleName,
  const tarch::la::Vector<Dimensions, double>& particleX,
  const int                                    particleID,
  bool                                         isLocal,
  const tarch::la::Vector<Dimensions, double>& vertexX,
  const tarch::la::Vector<Dimensions, double>& vertexH,
  int                                          treeId,
  const std::string&                           trace
) {}

void toolbox::particles::assignmentchecks::assignParticleToSieveSet(
  const std::string&                           particleName,
  const tarch::la::Vector<Dimensions, double>& particleX,
  const int                                    particleID,
  bool                                         isLocal,
  const tarch::la::Vector<Dimensions, double>& vertexH,
  int                                          treeId,
  const std::string&                           trace
) {}

void toolbox::particles::assignmentchecks::moveParticle(
  const std::string&                           particleName,
  const tarch::la::Vector<Dimensions, double>& oldParticleX,
  const tarch::la::Vector<Dimensions, double>& newParticleX,
  const int                                    particleID,
  const tarch::la::Vector<Dimensions, double>& vertexX,
  const tarch::la::Vector<Dimensions, double>& vertexH,
  int                                          treeId,
  const std::string&                           trace
) {}

void toolbox::particles::assignmentchecks::ensureDatabaseIsEmpty() {}

toolbox::particles::assignmentchecks::internal::Event toolbox::particles::assignmentchecks::internal::getPreviousEvent(ParticleEvents& particleHistory, int spacetreeId, size_t nFirstEventsToSkip) {
  return Event(Event::Type::NotFound);
}
#endif
