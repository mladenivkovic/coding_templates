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
// std::cout <<
//       "addEvent(...) " <<
//       "add new particle history thread for "
//         << identifier.toString()
// << std::endl;
    // );
    return _data.at(identifier);

  } else {

    ParticleEvents& history = search->second;

    // First: Do we need to downsize the history?
    if (history.size() >= _maxParticleSnapshotsToKeepTrackOf){

      Event last = history.back();

      // TODO: Re-Insert
      // logDebug(
// std::cout <<
//         "addEvent(...) " <<
//         "shortening history to " << last.toString()
// << std::endl;
      // );

      last.trace = "substitute-for-whole-trajectory/" + last.trace;
      history.clear();
      history.push_back(last);
    }

    // Add the new event now.
    history.push_back(event);

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
// std::cout <<
//         "addEvent(...) " <<
//         "shifting particle identifier from "
//           << key.particleX << " to " << identifier.particleX
// << std::endl;
      // );

      ParticleEvents historyCopy = history;
      _data.erase(key);
      _data.insert(std::pair<ParticleIdentifier, ParticleEvents>( identifier, historyCopy));
    }

    return history;
  }
}


toolbox::particles::assignmentchecks::internal::Event toolbox::particles::assignmentchecks::internal::getPreviousEvent(ParticleEvents& particleHistory, int spacetreeId, size_t nFirstEventsToSkip) {

  if (particleHistory.size() <= nFirstEventsToSkip){
    // TODO: Put back in
    // logWarning(
std::cout << "Searching in empty particleHistory " <<
  particleHistory.size() <<
  nFirstEventsToSkip
<< std::endl;
        // );
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

  // TODO: exception handle shortened history. We may have deleted the event.

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


void toolbox::particles::assignmentchecks::eraseParticle(
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
  // logTraceInWith4Arguments(
  //   "eraseParticle(...)",
  //   particleName,
  //   particleX,
  //   isLocal,
  //   treeId
  // );


  /*   internal::ParticleSearchIdentifier identifier = _database.createParticleSearchIdentifier( */
  /*   particleName, */
  /*   particleX, */
  /*   particleID, */
  /*   tarch::la::max(vertexH) */
  /* ); */
  /* internal::Event event(internal::Event::Type::Erase, isLocal, treeId, trace); */
  /*  */
  /* internal::Event previousLocalParticle = _database.getEntry(identifier, treeId, identifier.positionTolerance, identifier.positionTolerance) */
  /*                                           .first; */
  /* assertion5( */
  /*   previousLocalParticle.type == internal::Event::Type::DetachFromVertex, */
  /*   identifier.toString(), */
  /*   event.toString(), */
  /*   previousLocalParticle.toString(), */
  /*   treeId, */
  /*   _database.particleHistory(identifier) */
  /* ); */
  /*  */
  /* _database.addEvent(identifier, event); */
    // TODO: Re-Insert
  // logTraceOut("eraseParticle(...)");
}


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

  if (particleIsNew and (not checkNewParticles)){
    _database.addEvent(identifier, event);
  }

  else {
    ParticleEvents& history = _database.getParticleHistory(identifier);
    Event previousEvent = getPreviousEvent(history, treeId);
    _database.addEvent(identifier, event);

    const bool isDropping = previousEvent.type == internal::Event::Type::DetachFromVertex
                            and tarch::la::allGreater(previousEvent.vertexH, vertexH);
    const bool isLifting = previousEvent.type == internal::Event::Type::DetachFromVertex
                           and tarch::la::allSmaller(previousEvent.vertexH, vertexH);
    const bool isDroppingFromSieveSet = previousEvent.type == internal::Event::Type::AssignToSieveSet;

    const bool isDetached = reassignmentOnSameTreeDepthAllowed ?
        (previousEvent.type == internal::Event::Type::DetachFromVertex) :
        (previousEvent.type == internal::Event::Type::DetachFromVertex and not previousEvent.isLocal);

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
          (previousEvent.type == internal::Event::Type::DetachFromVertex and
          previousEvent.isLocal),
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

  assert(false);
    // TODO: Re-Insert
  // logTraceInWith3Arguments(
  //   "moveParticle(...)",
  //   particleName,
  //   oldParticleX,
  //   newParticleX
  // );

  // we use this to correctly ID a particle
  // const double idSearchTolerance = tarch::la::max(vertexH);
  // we use this as leeway to find a particle in the past.
  // Divide py ::Precision because it will be multiplied again in numericalEquals().
  // We want to be able to search the entire vertex size through the past.
  // const double pastSearchTolerance = tarch::la::max(vertexH) / internal::ParticleSearchIdentifier::Precision;
  // const double dx_old = tarch::la::norm2(newParticleX - oldParticleX);

  // TODO: THIS IS VERY VERY WRONG
  // const double minDx = 1.;


  // internal::ParticleSearchIdentifier newIdentifier = _database.createParticleSearchIdentifier(
  //   particleName,
  //   newParticleX,
  //   particleID,
  //   idSearchTolerance
  // );

  // auto previousEntry = _database.getEntry(oldIdentifier, treeId, internal::ParticleSearchIdentifier::getMinDx(), pastSearchTolerance);
  // internal::Event previousEvent = previousEntry.first;
  // internal::ParticleSearchIdentifier previousIdentifier = previousEntry.second;



  // Find the last recorded particle position. Since we're moving particles here,
  // they must've been assigned to a vertex in the past, so at least 1 event must exist.
  // tarch::la::Vector<Dimensions, double> previousParticleX =
  //   _database.getPreviousParticlePosition(newIdentifier, treeId, minDx, pastSearchTolerance);

  // assertion(previousEvent.type != internal::Event::Type::NotFound);
  // assertion(
  //     previousEvent.type == internal::Event::Type::AssignToVertex
  // );



  // std::cout << "\n\tNEW " << newIdentifier.particleX <<
  //   "\n\tPREV " << previousParticleX <<
  //   "\n\tDIFF " << newParticleX - previousParticleX <<
  //   "\n\tTOLERANCE " << idSearchTolerance * internal::ParticleSearchIdentifier::Precision <<
  //   std::endl;
  //
  //   const double dx_since_last_entry = tarch::la::norm2(newParticleX - previousParticleX);
  //


    // First check: Are we even close enough for our set precision limit?
    // Second check: Do we want to trace this particle's motion?
    /* if (//not (newIdentifier.numericalEquals(previousIdentifier)) and */
  /*       dx_since_last_entry >= idSearchTolerance * internal::ParticleSearchIdentifier::Precision */
  /*       // not tarch::la::equals( */
  /*       //   newParticleX, */
  /*       //   previousEvent.previousParticleX, */
  /*       //   idSearchTolerance) */
  /*     ) { */
  /*  */
  /*  */
  /*       internal::ParticleSearchIdentifier oldIdentifier = _database.createParticleSearchIdentifier( */
  /*         particleName, */
  /*         oldParticleX, */
  /*         particleID, */
  /*         0.5 * minDx, */
  /*         0.5 * dx_since_last_entry */
  /*         // idSearchTolerance */
  /*         // pastSearchTolerance */
  /*       ); */
  /*  */
  /*  */
  /* std::cout << "\n\n HELLO THEREEEEEEEEEEEEEEEEEE \n\n"; */
  /*  */
  /*       internal::Event newEvent( */
  /*         internal::Event::Type::MoveWhileAssociatedToVertex, */
  /*         vertexX, */
  /*         newIdentifier.particleX, */
  /*         vertexH, */
  /*         treeId, */
  /*         trace */
  /*       ); */
  /*  */
  /*       internal::Event previousEvent */
  /*         = _database.getEntry( */
  /*             newIdentifier, */
  /*             internal::Database::AnyTree, */
  /*             0.5 * minDx , */
  /*             // 0.5 * dx_since_last_entry / internal::ParticleSearchIdentifier::Precision).first; */
  /*             dx_since_last_entry / internal::ParticleSearchIdentifier::Precision).first; */
  /*       internal::Event existingNewEventOnAnyTree */
  /*         = _database.getEntry( */
  /*             newIdentifier, */
  /*             internal::Database::AnyTree, */
  /*             minDx , */
  /*             minDx).first; */
  /*       internal::Event existingNewEventOnLocalTree */
  /*         = _database.getEntry( */
  /*             newIdentifier, */
  /*             treeId, */
  /*             minDx , */
  /*             minDx).first; */
  /*  */
  /*       const std::string errorMessage0 = R"( */
  /* ============= */
  /* Explanation */
  /* ============= */
  /* The tracer has been informed of a particle movement. When it tried to bookmark */
  /* the particle with its new position, it found out that there is already a */
  /* particle registered at this place. It seems that a particle overlaps with */
  /* another one. */
  /*  */
  /* This might mean that there is actually a particle here, but could also result */
  /* from two other situations: */
  /*  */
  /* - We trace position updates one after the other. If particle A takes the */
  /*   position of a particle B, we might simply not have updated B yet. */
  /* - We trace position updates only if positions have changed significantly. */
  /*   Significantly here is formalised via */
  /*  */
  /*       toolbox::particles::assignmentchecks::internal::ParticleSearchIdentifier::Precision */
  /*  */
  /*   That is, if particles are closer together than this delta, we do not write */
  /*   logs into our database. This ensures that the database is not filled with */
  /*   tiny update entries. */
  /*  */
  /* As the tracing cannot handle either situation, we are left with two options. */
  /* We can dramatically reduce Precision at the cost of a higher overhead. */
  /* Alternatively, it might be appropriate to check the time step sizes */
  /* employed: If particles move too fast, the probability that A ends up at a */
  /* position just previously held by B (which is not yet updated) is higher. */
  /*  */
  /* )"; */
  /*       assertion13( */
  /*         existingNewEventOnLocalTree.type == internal::Event::Type::NotFound, */
  /*         newIdentifier.toString(), */
  /*         oldIdentifier.toString(), */
  /*         previousEvent.toString(), */
  /*         newEvent.toString(), */
  /*         existingNewEventOnLocalTree.toString(), */
  /*         existingNewEventOnAnyTree.toString(), */
  /*         _database.getNumberOfSnapshots(), */
  /*         treeId, */
  /*         trace, */
  /*         internal::ParticleSearchIdentifier::Precision, */
  /*         _database.totalEntries(newIdentifier), */
  /*         _database.particleHistory(newIdentifier), */
  /*         errorMessage0 */
  /*       ); */
  /*       const std::string errorMessage1 = R"( */
  /* ============= */
  /* Explanation */
  /* ============= */
  /* The tracer has been informed of a particle movement. When it tried to bookmark */
  /* the particle with its new position, it found out that there is already a */
  /* particle registered at this place. That is fine, as particles might be held */
  /* redundantly on different trees - either as halo copies or as they sit exactly */
  /* on the face between two subdomains. */
  /*  */
  /* If that happens, they however have to be tied to the same vertex in the domain */
  /* although the vertex might be replicated on a different tree. Alternatively, the */
  /* other rank might already have moved it and come to the conclusion that it has */
  /* to be assigned to the sieve set. The present tree is not there yet, i.e. is */
  /* just about to move it, but will eventually also raise its particle to the */
  /* sieve set. */
  /* )"; */
  /*       assertion13( */
  /*         existingNewEventOnAnyTree.type == internal::Event::Type::NotFound */
  /*         or */
  /*         existingNewEventOnAnyTree.type == internal::Event::Type::AssignToSieveSet */
  /*         or */
  /*         ( */
  /*           existingNewEventOnAnyTree.type == internal::Event::Type::AssignToVertex */
  /*           and */
  /*           existingNewEventOnAnyTree.vertexX == previousEvent.vertexX */
  /*           and */
  /*           existingNewEventOnAnyTree.vertexH == previousEvent.vertexH */
  /*         ), */
  /*         oldIdentifier.toString(), */
  /*         newIdentifier.toString(), */
  /*         previousEvent.toString(), */
  /*         newEvent.toString(), */
  /*         existingNewEventOnLocalTree.toString(), */
  /*         existingNewEventOnAnyTree.toString(), */
  /*         _database.getNumberOfSnapshots(), */
  /*         treeId, */
  /*         trace, */
  /*         internal::ParticleSearchIdentifier::Precision, */
  /*         _database.totalEntries(newIdentifier), */
  /*         _database.particleHistory(newIdentifier), */
  /*         errorMessage1 */
  /*       ); */
  /*       assertion12( */
  /*         previousEvent.type == internal::Event::Type::AssignToVertex, */
  /*         // or previousEvent.type == internal::Event::Type::MoveWhileAssociatedToVertex, */
  /*         oldIdentifier.toString(), */
  /*         previousEvent.toString(), */
  /*         newIdentifier.toString(), */
  /*         newEvent.toString(), */
  /*         _database.getNumberOfSnapshots(), */
  /*         treeId, */
  /*         trace, */
  /*         _database.totalEntries(oldIdentifier), */
  /*         _database.particleHistory(oldIdentifier), */
  /*         _database.totalEntries(newIdentifier), */
  /*         _database.particleHistory(newIdentifier), */
  /*         _database.toString() */
  /*       ); */
  /*  */
  /* std::cout << "     ADDING MOVE EVENT" << std::endl; */
  /*       _database.addEvent(newIdentifier, newEvent); */
  /*     } */
  /*   // } */
  /*  */
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

  assert(false);
  // TODO: Re-Insert
  /* logTraceInWith6Arguments( */
    /* "detachParticleFromVertex(...)", */
    /* particleName, */
    /* particleX, */
    /* isLocal, */
    /* vertexX, */
    /* vertexH, */
    /* treeId */
  /* ) */;

  /* internal::ParticleSearchIdentifier identifier = _database.createParticleSearchIdentifier( */
  /*   particleName, */
  /*   particleX, */
  /*   particleID, */
  /*   tarch::la::max(vertexH) */
  /* ); */
  /* // TODO MLADEN: CHECK *3 above */
  /* internal::Event event{ */
  /*   internal::Event::Type::DetachFromVertex, */
  /*   isLocal, */
  /*   vertexX, */
  /*   particleX, */
  /*   vertexH, */
  /*   treeId, */
  /*   trace}; */
  /*  */
  /* // tarch::la::Vector<Dimensions, double> previousParticleX = */
  /* //   _database.getPreviousParticlePosition(newIdentifier, treeId, minDx, pastSearchTolerance); */
  /*  */
  /* internal::Event previousEvent = _database.getEntry( */
  /*     identifier, */
  /*     treeId, */
  /*     // identifier.positionTolerance, */
  /*     // internal::ParticleSearchIdentifier::getMinDx() , */
  /*     0.1, */
  /*     identifier.positionTolerance).first; */
  /*  */
  /* assertion8( */
  /*   previousEvent.type == internal::Event::Type::AssignToVertex, */
  /*   // or previousEvent.type == internal::Event::Type::MoveWhileAssociatedToVertex, */
  /*   identifier.toString(), */
  /*   event.toString(), */
  /*   previousEvent.toString(), */
  /*   treeId, */
  /*   _database.getNumberOfSnapshots(), */
  /*   trace, */
  /*   _database.particleHistory(identifier), */
  /*   _database.toString() */
  /* ); */
  /* assertion7( */
  /*   tarch::la::equals(previousEvent.vertexX, vertexX), */
  /*   identifier.toString(), */
  /*   event.toString(), */
  /*   previousEvent.toString(), */
  /*   treeId, */
  /*   _database.getNumberOfSnapshots(), */
  /*   trace, */
  /*   _database.particleHistory(identifier) */
  /* ); */
  /* assertion6( */
  /*   tarch::la::equals(previousEvent.vertexH, vertexH), */
  /*   identifier.toString(), */
  /*   event.toString(), */
  /*   previousEvent.toString(), */
  /*   treeId, */
  /*   trace, */
  /*   _database.particleHistory(identifier) */
  /* ); */
  /*  */
  /* _database.addEvent(identifier, event); */
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
