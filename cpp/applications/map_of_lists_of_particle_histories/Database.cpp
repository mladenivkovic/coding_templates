#include "Database.h"
#include "ParticleIdentifier.h"
#include "Event.h"

#include <iterator>
#include <limits>
#include <ranges>
#include <vector>

// #include "tarch/multicore/MultiReadSingleWriteLock.h"


#if !defined(AssignmentChecks) and !defined(noAssignmentChecks) \
  and PeanoDebug > 0
#define AssignmentChecks
#endif

// TODO: Add this back
// namespace {
//   tarch::logging::Log _log("toolbox::particles::assignmentchecks");
//
//   toolbox::particles::assignmentchecks::internal::Database _database;
// } // namespace

namespace {
  // tarch::logging::Log _log("toolbox::particles::assignmentchecks");

  toolbox::particles::assignmentchecks::internal::Database _database;
} // namespace



// toolbox::particles::assignmentchecks::internal::ParticleIdentifier toolbox::particles::assignmentchecks::internal::Database::createParticleIdentifier(
//     const std::string&                           particleName,
//     const tarch::la::Vector<Dimensions, double>& particleX,
//     const int                                    particleID,
//     const double                                 idSearchTolerance,
//     const double                                 pastSearchTolerance
//   ) {
//
  // ParticleIdentifier result(particleName, particleX, particleID, idSearchTolerance);
  /* // need a different tolerance for the past event/record search */
  /* ParticleIdentifier */
  /*   pastSearchID(particleName, particleX, particleID, pastSearchTolerance); */
  /*  */
  /* if (pastSearchTolerance > 0.0) { */
  /*   // tarch::multicore::MultiReadSingleWriteLock */
  /*   //      lock(_semaphore, tarch::multicore::MultiReadSingleWriteLock::Read); */
  /*   auto currentSnapshot = _data.crbegin(); */
  /*   while (currentSnapshot != _data.crend()) { */
  /*     for (const auto& eventsForOneParticle : *currentSnapshot) { */
  /*       // use floating-point aware comparison operator */
  /*       // if (eventsForOneParticle.first == pastSearchID) { */
  /*       // if (pastSearchID.positionStrictNumericalEquals(eventsForOneParticle.first)) { */
  /*       if (pastSearchID == eventsForOneParticle.first) { */
  /*         // TODO: re-insert */
  /*         // logDebug( */
  /*         //   "createParticleIdentifier()", */
  /*           // "found entry for " */
  /*           //   << particleX << " given tolerance of " << pastSearchTolerance */
  /*           //   << ": will copy data over bit-wisely which biases identifier by " */
  /*           //   << (eventsForOneParticle.first.particleX - pastSearchID.particleX)  << "\n ID is " << result.toString() */
  /*           // ); */
  /*         // This is a bit-wise copy and biases the result towards an */
  /*         // existing entry. */
  /*         result = eventsForOneParticle.first; */
  /*         assertion(currentSnapshot->count(result) > 0); */
  /*         return result; */
  /*       } */
  /*     } */
  /*     currentSnapshot++; */
  /*   } */
  /* } */
/*  */
//   return result;
// }


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


void toolbox::particles::assignmentchecks::internal::Database::
  eliminateExistingParticles() {
  // tarch::multicore::MultiReadSingleWriteLock
  //   lock(_semaphore, tarch::multicore::MultiReadSingleWriteLock::Write);

  // bool hasEliminated = true;
  //
  // while (hasEliminated) {
  //   removeEmptyDatabaseSnapshots();
  //
  //   hasEliminated     = false;
  //   auto lastSnapshot = _data.rbegin();
  //
  //   auto particleTrajectory = lastSnapshot->begin();
  //   while (particleTrajectory != lastSnapshot->end() and not hasEliminated) {
  //     if (
  //       particleTrajectory->second.back().type == Event::Type::MoveWhileAssociatedToVertex
  //       or
  //       particleTrajectory->second.back().type == Event::Type::AssignToVertex
  //       or
  //       not particleTrajectory->second.back().isLocal
  //     ) {
  //       removeTrajectory(
  //         particleTrajectory->first,
  //         particleTrajectory->second.back().treeId
  //       );
  //       hasEliminated = true;
  //     } else {
  //       particleTrajectory++;
  //     }
  //   }
  //
  //   if (lastSnapshot->empty()) {
  //     _data.pop_back();
  //     hasEliminated = true;
  //   }
  // }
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

void toolbox::particles::assignmentchecks::internal::Database::
  removeEmptyDatabaseSnapshots() {
/*   auto snapshot        = _data.begin(); */
/*   int  currentSnapshot = 0; */
/*   while (snapshot != _data.end()) { */
/*     auto trajectory = snapshot->begin(); */
/*     while (trajectory != snapshot->end()) { */
/*       if (trajectory->second.empty()) { */
/* // TODO: re-insert */
/*         // logDebug( */
/*         //   "removeEmptyDatabaseSnapshots()", */
/*         //   "removed entry for particle " << trajectory->first.toString() */
/*         // ); */
/*         trajectory = snapshot->erase(trajectory); */
/*       } else { */
/*         trajectory++; */
/*       } */
/*     } */
/*     snapshot++; */
/*   } */
/*  */
/*  */
/*   snapshot       = _data.begin(); */
/*   const int size = _data.size(); */
/*   while (snapshot != _data.end()) { */
/*     if (snapshot->empty() and currentSnapshot < size - 1) { */
/*       // TODO: re-insert */
/*       // logDebug( */
/*       //   "removeEmptyDatabaseSnapshots()", */
/*       //   "removed whole snapshot as it was empty" */
/*       // ); */
/*       snapshot = _data.erase(snapshot); */
/*     } else { */
/*       snapshot++; */
/*     } */
/*     currentSnapshot++; */
/*   } */
}


void toolbox::particles::assignmentchecks::internal::Database::removeTrajectory(
  const ParticleSearchIdentifier& identifier,
  int                       spacetreeId,
  int                       firstNRecentEntriesToSkip
) {
  assertion(spacetreeId >= 0);

  // auto currentSnapshot = _data.rbegin();
  /* std::advance(currentSnapshot, firstNRecentEntriesToSkip); */
  /*  */
  /* while (currentSnapshot != _data.rend()) { */
  /*   MeshSweepData& meshSweepData = *currentSnapshot; */
  /*  */
  /*   if (meshSweepData.count(identifier) > 0) { */
  /*     auto historyEventIterator = meshSweepData.at(identifier).rbegin(); */
  /*     while (historyEventIterator != meshSweepData.at(identifier).rend()) { */
  /*       if (historyEventIterator->treeId == spacetreeId and historyEventIterator->type == Event::Type::MoveWhileAssociatedToVertex) { */
  /*         ParticleSearchIdentifier previousIdentifier = identifier; */
  /*         previousIdentifier.particleX = historyEventIterator->previousParticleX; */
  /*         // TODO: Re-Insert */
  /*         // logDebug( */
  /*         //   "removeTrajectory(...)", */
  /*         //   "first erase historic data of " << previousIdentifier.toString( */
  /*         //   ) << " due to " << historyEventIterator->toString() */
  /*         // ); */
  /*         removeTrajectory( */
  /*           previousIdentifier, */
  /*           spacetreeId, */
  /*           firstNRecentEntriesToSkip */
  /*         ); */
  /*       } */
  /*       historyEventIterator++; */
  /*     } */
  /*  */
  /*     auto forwardEventIterator = meshSweepData.at(identifier).begin(); */
  /*     while (forwardEventIterator != meshSweepData.at(identifier).end()) { */
  /*       if (forwardEventIterator->treeId == spacetreeId) { */
  /*         // TODO: Re-Insert */
  /*         // logDebug( */
  /*         //   "removeTrajectory(...)", */
  /*         //   "erase event " << forwardEventIterator->toString() */
  /*         // ); */
  /*         forwardEventIterator = meshSweepData[identifier].erase( */
  /*           forwardEventIterator */
  /*         ); */
  /*       } else { */
  /*         forwardEventIterator++; */
  /*       } */
  /*     } */
  /*   } */
  /*   currentSnapshot++; */
  /*   firstNRecentEntriesToSkip++; */
  /* } */
}


std::pair<
  toolbox::particles::assignmentchecks::internal::Event,
  toolbox::particles::assignmentchecks::internal::ParticleIdentifier>
  toolbox::particles::assignmentchecks::internal::Database::getEntry(
    const ParticleSearchIdentifier& identifier,
    const int                       spacetreeId,
    const double                    idSearchTolerance,
    const double                    pastSearchTolerance,
    int                             firstNRecentSweepsToSkip
  ) {
  // tarch::multicore::MultiReadSingleWriteLock
  //   lock(_semaphore, tarch::multicore::MultiReadSingleWriteLock::Read);

/*   auto currentSnapshot = _data.crbegin(); */
/*  */
/*   const ParticleSearchIdentifier pastSearchIdentifier = ParticleSearchIdentifier( */
/*       // identifier.particleName, identifier.particleX, identifier.particleID, idSearchTolerance */
/*       identifier.particleName, identifier.particleX, identifier.particleID, pastSearchTolerance */
/*       ); */
/*  */
/*   std::advance(currentSnapshot, firstNRecentSweepsToSkip); */
/*  */
/*   // Try all recorded snapshots. */
/*   while (currentSnapshot != _data.crend()) { */
/*     const MeshSweepData& meshSweepData = *currentSnapshot; */
/*  */
/*     // Do we have a record of this particle in the past? */
/*     if (meshSweepData.count(pastSearchIdentifier) > 0) { */
/*       auto event = meshSweepData.at(pastSearchIdentifier).crbegin(); */
/*  */
/*       while (event != meshSweepData.at(pastSearchIdentifier).crend()) { */
/*         bool treeIsAFit = event->treeId == spacetreeId or spacetreeId == AnyTree; */
/*  */
/*         if (event->type == Event::Type::Erase and treeIsAFit) { */
/*           // return the identifier with the correct tolerance. */
/*           std::cout << " GET_ENTRY EXIT 1 "<< std::endl; */
/*           return {Event(Event::Type::NotFound), identifier}; */
/*         } */
/*         else if (event->type == Event::Type::MoveWhileAssociatedToVertex and treeIsAFit and firstNRecentSweepsToSkip != DoNotFollowParticleMovementsInDatabase) { */
/*  */
/* std::cout << "\n\tCHECKING EVENT " << */
/*               event->previousParticleX << */
/*               " TYPE " << */
/*               static_cast<int>(event->type) << */
/*               " STR " << */
/*               event->toString() << */
/*               " IDSEARCHTOL " << idSearchTolerance << */
/*               " PASTSEARCHTOL " << pastSearchTolerance << */
/*               " PARTX " << identifier.particleX << std::endl; */
/*  */
/*           const ParticleSearchIdentifier previousIdentifier */
/*             = _database.createParticleSearchIdentifier( */
/*               identifier.particleName, */
/*               event->previousParticleX, */
/*               identifier.particleID, */
/*               idSearchTolerance, */
/*               pastSearchTolerance */
/*             ); */
/*  */
/*           // _database.createParticleSearchIdentifier may return a previous identifier */
/*           // with a different, less strict positionTolerance than we want here. */
/*           // This may lead to wrong positives. We want a strict(er) one, so check */
/*           // for that as well. */
/*           assertion3( */
/*             not( previousIdentifier == identifier ) */
/*               or (tarch::la::norm2(previousIdentifier.particleX - identifier.particleX) > idSearchTolerance), */
/*             previousIdentifier.toString(), */
/*             identifier.toString(), */
/*             event->toString() */
/*           ); */
/*  */
/*           // TODO: Re-Insert */
/*           // logDebug( */
/*           //   "getEntry()", */
/*           //   "rerun with " */
/*           //     << previousIdentifier.toString() << " distilled from " */
/*           //     << identifier.toString() << " on iteration " */
/*           //     << firstNRecentSweepsToSkip */
/*           // ); */
/*           std::cout << " GET_ENTRY EXIT 2"<< std::endl; */
/*  */
/*           return {*event, previousIdentifier}; */
/*  */
/*           // return getEntry( */
/*           //   previousIdentifier, */
/*           //   spacetreeId, */
/*           //   idSearchTolerance, */
/*           //   idSearchTolerance, */
/*           //   firstNRecentSweepsToSkip */
/*             // ); */
/*         } */
/*         else if (event->type != Event::Type::Erase and treeIsAFit) { */
/*           std::cout << " GET_ENTRY EXIT 3=" << std::endl; */
/*           return {*event, identifier}; */
/*         } */
/*         event++; */
/*       } */
/*     } */
/*     currentSnapshot++; */
/*     firstNRecentSweepsToSkip++; */
/*   } */
/*  */
/*   std::cout << " GET_ENTRY EXIT 4 " << std::endl; */
  return {Event(Event::Type::NotFound), identifier};
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
    << std::endl
    << "============================" << std::endl
    << identifier.toString() << std::endl
    << "============================";

  auto search = _data.find(identifier);
  if (search != _data.end()){

    ParticleEvents& history = search->second;
    auto ev = history.crbegin();

    // mesh sweep index of the previous event.
    int prevMeshSweepInd = -1;

    while (ev != history.crend()){

      int meshSweepInd = ev->meshSweepIndex;

      if (meshSweepInd == prevMeshSweepInd){
        // next event during the same sweep
        msg << "->" << ev->toString() << "\n\t";
      } else {
        // We're starting new sweep. Print header.
        msg << std::endl << "sweep #" << meshSweepInd << " (" << _meshSweepData.at(meshSweepInd).getName() << "):\n\t";
      }

      prevMeshSweepInd = meshSweepInd;
      ev++;
    }
  }

  return msg.str();
}

// checked
void toolbox::particles::assignmentchecks::internal::Database::addEvent(
  ParticleSearchIdentifier identifier,
  Event&       event
) {
  // tarch::multicore::MultiReadSingleWriteLock
  //   lock(_semaphore, tarch::multicore::MultiReadSingleWriteLock::Write);
  //

  // Take note of the current mesh sweep index.
  event.meshSweepIndex = _database.getCurrentMeshSweepIndex();

  int count = _data.count(identifier) ;
  if (count == 0){
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

  } else {

    ParticleEvents& history = _data[identifier];
    history.push_back(event);

std::cout <<
      "addEvent(...)" <<
      " add new EVENT for "
        << identifier.toString()
        << " "
        << event.toString()
<< std::endl;

  }

  // assertion(not _data.empty());
  /* MeshSweepData& snapshot = *_data.rbegin(); */
  /* if (snapshot.count(identifier) == 0) { */
  /*   snapshot.insert(std::pair<ParticleSearchIdentifier, ParticleEvents>( */
  /*     identifier, */
  /*     ParticleEvents() */
  /*   )); */
  /*   // TODO: Re-Insert */
  /*   // logDebug( */
  /*   //   "addEvent(...)", */
  /*   //   "add new particle history thread in this snapshot for " */
  /*   //     << identifier.toString() */
  /*   // ); */
  /* } */
  /*  */
  /* // We first have to push it. Otherwise, the susequent getEntry() won't work. */
  /* snapshot[identifier].push_back(event); */
  /*  */
  /* if (event.type == Event::Type::AssignToVertex and _data.size() > _maxParticleSnapshotsToKeepTrackOf) { */
  /*   removeTrajectory(identifier, event.treeId); */
  /*   removeEmptyDatabaseSnapshots(); */
  /*  */
  /*   // re-add element */
  /*   if (snapshot.count(identifier) == 0) { */
  /*     snapshot.insert(std::pair<ParticleSearchIdentifier, ParticleEvents>( */
  /*       identifier, */
  /*       ParticleEvents() */
  /*     )); */
  /*   // TODO: Re-Insert */
  /*     // logDebug( */
  /*     //   "addEvent(...)", */
  /*     //   "re-add particle history in this snapshot for " << identifier.toString() */
  /*     // ); */
  /*   } */
  /*   snapshot[identifier].push_back(event); */
  /* } */
  /* if (event.type == Event::Type::MoveWhileAssociatedToVertex and _data.size() > _maxParticleSnapshotsToKeepTrackOf) { */
  /*   // TODO: Re-Insert */
  /*   // lock.free(); */
  /*  */
  /*   // THIS IS WRONG; USED TO HAVE MINDX instead of 0.1 */
  /*   auto rootEntryOfLatestTrajectory = getEntry(identifier, event.treeId, 0.1, identifier.positionTolerance); */
  /*   assertion4( */
  /*     rootEntryOfLatestTrajectory.first.type != Event::Type::NotFound, */
  /*     rootEntryOfLatestTrajectory.first.toString(), */
  /*     rootEntryOfLatestTrajectory.second.toString(), */
  /*     event.toString(), */
  /*     identifier.toString() */
  /*   ); */
  /*  */
  /*   Event substituteEntryForTrajectory( */
  /*     Event::Type::MoveWhileAssociatedToVertex, */
  /*     rootEntryOfLatestTrajectory.first.vertexX, */
  /*     rootEntryOfLatestTrajectory.second.particleX, */
  /*     rootEntryOfLatestTrajectory.first.vertexH, */
  /*     event.treeId, */
  /*     "substitute-for-whole-trajectory" */
  /*   ); */
  /*   rootEntryOfLatestTrajectory.first.trace */
  /*     = "substitute-trajectory-start-from-original-point-" */
  /*       + ::toString(rootEntryOfLatestTrajectory.second.particleX); */
  /*  */
  /*  */
  /*   // TODO: Re-Insert */
  /*   // lock.lock(); */
  /*   removeTrajectory(identifier, event.treeId); */
  /*   removeEmptyDatabaseSnapshots(); */
  /*  */
  /*   // re-add element */
  /*   if (snapshot.count(identifier) == 0) { */
  /*     snapshot.insert(std::pair<ParticleSearchIdentifier, ParticleEvents>( */
  /*       identifier, */
  /*       ParticleEvents() */
  /*     )); */
  /*     // TODO: Re-Insert */
  /*     // logDebug( */
  /*     //   "addEvent(...)", */
  /*     //   "re-add particle history in this snapshot for " << identifier.toString() */
  /*     // ); */
  /*   } */
  /*   snapshot[rootEntryOfLatestTrajectory.second].push_back( */
  /*     rootEntryOfLatestTrajectory.first */
  /*   ); */
  /*   snapshot[identifier].push_back(substituteEntryForTrajectory); */
  /* } */
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

  /* constexpr bool checkNewParticles = false; */
  /*  */
  /* if ((not checkNewParticles) and particleIsNew) { */
  /*   internal::ParticleSearchIdentifier identifier = _database.createParticleSearchIdentifier( */
  /*     particleName, */
  /*     particleX, */
  /*     particleID, */
  /*     tarch::la::max(vertexH) */
  /*   ); */
  /*   internal::Event event( */
  /*     internal::Event::Type::AssignToVertex, */
  /*     isLocal, */
  /*     vertexX, */
  /*     particleX, */
  /*     vertexH, */
  /*     treeId, */
  /*     trace */
  /*   ); */
  /*  */
  /*   _database.addEvent(identifier, event); */
  /*  */
  /* } else { */
  /*  */
  /*   internal::ParticleSearchIdentifier identifier = _database.createParticleSearchIdentifier( */
  /*     particleName, */
  /*     particleX, */
  /*     particleID, */
  /*     tarch::la::max(vertexH) */
  /*   ); */
  /*   internal::Event event( */
  /*     internal::Event::Type::AssignToVertex, */
  /*     isLocal, */
  /*     vertexX, */
  /*     particleX, */
  /*     vertexH, */
  /*     treeId, */
  /*     trace */
  /*   ); */
  /*  */
  /*   internal::Event previousEvent = _database.getEntry(identifier, treeId, identifier.positionTolerance, identifier.positionTolerance).first; */
  /*  */
  /*   const bool isDropping = previousEvent.type == internal::Event::Type::DetachFromVertex */
  /*                           and tarch::la::allGreater(previousEvent.vertexH, vertexH); */
  /*   const bool isLifting = previousEvent.type == internal::Event::Type::DetachFromVertex */
  /*                          and tarch::la::allSmaller(previousEvent.vertexH, vertexH); */
  /*   const bool isDroppingFromSieveSet = previousEvent.type == internal::Event::Type::AssignToSieveSet; */
  /*  */
  /*   const bool isDetached = reassignmentOnSameTreeDepthAllowed ? */
  /*       (previousEvent.type == internal::Event::Type::DetachFromVertex) : */
  /*       (previousEvent.type == internal::Event::Type::DetachFromVertex and not previousEvent.isLocal); */
  /*  */
  /*   if (isLocal) { */
  /*     assertion7( */
  /*       previousEvent.type == internal::Event::Type::NotFound */
  /*       or */
  /*       isDroppingFromSieveSet */
  /*       or */
  /*       (isLifting and previousEvent.isLocal) */
  /*       or */
  /*       (isDropping and previousEvent.isLocal) */
  /*       or */
  /*       (isDetached), */
  /*       identifier.toString(), */
  /*       event.toString(), */
  /*       previousEvent.toString(), */
  /*       treeId, */
  /*       _database.getNumberOfSnapshots(), */
  /*       trace, */
  /*       _database.particleHistory(identifier) */
  /*     ); */
  /*   } else { */
  /*       assertion7( */
  /*         (previousEvent.type == internal::Event::Type::NotFound) */
  /*         or */
  /*         isDroppingFromSieveSet */
  /*         or */
  /*         (isDropping and not previousEvent.isLocal) */
  /*         or */
  /*         (previousEvent.type == internal::Event::Type::DetachFromVertex and */
  /*         previousEvent.isLocal), */
  /*         identifier.toString(), */
  /*         event.toString(), */
  /*         previousEvent.toString(), */
  /*         treeId, */
  /*         _database.getNumberOfSnapshots(), */
  /*         trace, */
  /*         _database.particleHistory(identifier) */
  /*       ); */
  /*   } */
  /*  */
  /*   _database.addEvent(identifier, event); */
  /* } */
  /*  */
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


void toolbox::particles::assignmentchecks::eliminateExistingParticles() {
  _database.eliminateExistingParticles();
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

void toolbox::particles::assignmentchecks::eliminateExistingParticles() {}

#endif
