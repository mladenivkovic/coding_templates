// This file is part of the Peano project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once


/* #include "peano4/datamanagement/CellMarker.h" */
/* #include "peano4/datamanagement/VertexEnumerator.h" */
#include "tarch/la/Vector.h"
/* #include "tarch/multicore/MultiReadSingleWriteSemaphore.h" */

/* #include "Database.h" */

namespace toolbox {
  namespace particles {
    namespace assignmentchecks {

      /**
       * Get history of stored sweeps.
       */
      std::string sweepHistory();

      /**
       * Inform API that this is the end of a grid run-throuch
       *
       * Delegates to the corresponding method in the database.
       */
      void startMeshSweep(const std::string& meshSweepName);

      /**
       * Log that a particle is erased
       *
       * This should solely happen for virtual particles, i.e. halo particles.
       * Before this function is called, the corresponding particle has to be
       * removed from its associated vertex.
       */
      void eraseParticle(
        const std::string&                           particleName,
        const tarch::la::Vector<Dimensions, double>& particleX,
        const int                                    particleID,
        bool                                         isLocal,
        const tarch::la::Vector<Dimensions, double>& vertexH,
        int                                          treeId,
        const std::string&                           trace
      );

      /**
       * Record the movement of a particle
       *
       * We insert a new entry which memorises the particle movements. However,
       * we don't want to hold arbitrary small wiggles. So we construct the old
       * and the new particle identifier (name + position) and if they are
       * different, subject the ParticleIdentifier comparison operator and its
       * precision, then we log the new event.
       * If oldParticleX however equals newParticleX, the function degenerates
       * to nop and nothing is traced.
       *
       * This means that we might skip a few (tiny) particle movements. So if
       * we finally search for the origin particle, we have to pick a bigger
       * tolerance. After all, we might have skipped quite a lot of moves. In
       * theory, it should be enough if we increased the tolerance by a factor
       * of two. In practice, this is still too restrictive. We did
       * make good experiences with increasing the tolerance by one order of
       * magnitude.
       *
       * As we work with this tolerances and therefore skip particles, we have
       * to be careful which old particle x we dump into the database. We
       * could use oldParticleX, but we have just discussed above that we
       * search for a corresponding entry which might be off slightly.
       * Therefore, we do not bookmark oldParticleX -> newParticleX, but
       * instead store found old particleX -> newParticleX.
       */
      void moveParticle(
        const std::string&                           particleName,
        const tarch::la::Vector<Dimensions, double>& oldParticleX,
        const tarch::la::Vector<Dimensions, double>& newParticleX,
        const int                                    particleID,
        const tarch::la::Vector<Dimensions, double>& vertexX,
        const tarch::la::Vector<Dimensions, double>& vertexH,
        int                                          treeId,
        const std::string&                           trace
      );

      /**
       * Assign a particle to a vertex
       *
       * Take the particle, perform some consistency checks, and then add the
       * particle event to our history database. The consistency checks differ
       * for local from virtual particles.
       *
       * ## Consistency checks
       *
       * ### Consistency checks for local particles
       *
       * - A new particle might have been inserted into the domain for the very
       *   first time. In this case, the particle has to be "free", i.e. it may
       *   not have been assigned to any other vertex yet. The search should
       *   reduce a NotFound.
       * - The particle might have been dropped from a coarser level. See
       *   corresponding discussion for virtual particles below including some
       *   sketches.
       * - The particle might have been lifted. Lifting can only occur for
       *   local particles, as it is a result of a move and we may only move
       *   local particles. If the assignment is due to a lift, this local
       *   particle has been assigned to a finer mesh resolution before.
       * - If we lift a particle into the global sieve set (as the particle
       *   tunnels or there's a horizontal cut through the spacetree, i.e. the
       *   coarser cell is held by another rank), we will sieve the particle
       *   into the local domain in the next step. In this case, the previous
       *   particle's state could be local or virtual: If the particle has moved
       *   and therefore ended up in the sieve set, we will not have updated
       *   its parallel flag yet. Therefore, it might have changed.
       * - A particle might change from remote to local. This can only happen
       *   in UpdateParallelState after a merge, which first detaches a particle
       *   and then readds it with the local flag set.
       *
       *
       * ### Consistency check for virtual particles
       *
       * - The particle comes in through the boundary as part of the boundary
       *   exchange. In this case, the particle has to be "free", i.e. it may
       *   not be assigned to any other vertex yet. The particle has to be new
       *   (NotFound in database).
       * - If it is an update of the halo copy, we would expect the
       *   corresponding merge() to replace this copy. A replacement manifests
       *   in an event sequence of detach->erase->assign, so we still get a
       *   NotFound throughout the assign.
       * - A virtual particle might also be subject of a mesh drop: If we lift
       *   a particle, exchange it along the boundary, we have to drop it on
       *   both sides again. We update its state
       *   ***after*** we have assigned it to a vertex, but we also set its
       *   state in the merge, so here the state has to be virtual.
       * - A local particle might have changed from local to virtual as it
       *   had moved. In this case, the update should detach the particle
       *   before it adds it again.
       * - We might drop due to a global sieve. In this case, the previous
       *   particle's state could be local or virtual: If the particle has moved
       *   and therefore ended up in the sieve set, we will not have updated
       *   its parallel flag yet. Therefore, it might have changed.
       *
       * In the scenario below, we illustrate the last scenario by means of an
       * interface between two ranks. It
       * is denoted below as grey partition (one left rank) and a blueish
       * partition (on the right rank). They have kind of a halo layer of
       * width one, which is a logical construct, i.e. not maintained manually.
       *
       * @image html scenario00_particle-through-domain-wall.png
       *
       * Study the central setup: In step N, yellow moves, while red will be
       * deleted after the traversal, as each step moving particles has to
       * update states as well (UpdateParallelState action set). Yellow now
       * might be lifted to the next level. So far, its state is local on the
       * grey rank, no matter if it stays within this subdomain or not. We
       * exchange the data after step N.
       *
       * In step N+1, either rank sets the state to local or virtual. This
       * happens either in touchVertexFirstTime() in the UpdateParallelState
       * action set, or it happens straight in merge(). No matter what, it
       * happens on the coarser level before we even touch the fine mesh
       * level. When we drop it, the particle has the correct parallel state.
       *
       *
       * Another special situation arises from a setup, where the particle of
       * interest is located exactly along the subdomain boundary. In this
       * case, we move both copies, as the particle is considered to be local
       * on both ranks. Let one rank (tree 0) subsequently lift the particle,
       * as it has moved quite a bit. The other rank (tree 1) therefore also
       * has to lift yet cannot do so, as the next coarser mesh level is
       * actually owned by tree 0. It adds its particle to the sieve set.
       * After the movement, the particle is on the coarser level on tree 0
       * and in the sieve set of tree 1.
       *
       * This sieve set is now exchanged between the trees. Tree 1 seems to be
       * kind of simple: The particle will be taken from the sieve set,
       * inserted into the domain (given that it hasn't totally left this
       * tree) and will then either be set to local or virtual. However, we
       * will also get a copy of this particle from tree 0 through the
       * boundary. This copy will then be sieved. We end up with two copies:
       * One from the global sieve set and one from the boundary.
       *
       * Tree 0's behaviour exhibits the same behaviour: We will drop the
       * particle which we have temporarily lifted to the next coarser level.
       * After we have dropped it, we will also insert the copy from the
       * global sieve set.
       *
       * @param particleIsNew If we know that a particle is new, there's no
       *                      need to search through the database.
       * @param reassignmentOnSameTreeDepthAllowed If true, assume our
       *                      particle sorting/bookkeeping algorithm allows
       *                      for a particle to be reassigned from one vertex
       *                      to another vertex on the same depth in the tree
       *                      directly. Otherwise, expect the particle to be
       *                      sorted either via sieves, lifts, or drops.
       */
      void assignParticleToVertex(
        const std::string&                           particleName,
        const tarch::la::Vector<Dimensions, double>& particleX,
        const int                                    particleID,
        bool                                         isLocal,
        const tarch::la::Vector<Dimensions, double>& vertexX,
        const tarch::la::Vector<Dimensions, double>& vertexH,
        int                                          treeId,
        const std::string&                           trace,
        bool                                         particleIsNew = false,
        bool                                         reassignmentOnSameTreeDepthAllowed = false
      );


      /**
       * Remove particle from vertex
       *
       * To make this API call valid, the particle has to be assigned to a
       * vertex beforehand. So there has to be a assignParticleToVertex()
       * before this one.
       */
      void detachParticleFromVertex(
        const std::string&                           particleName,
        const tarch::la::Vector<Dimensions, double>& particleX,
        const int                                    particleID,
        bool                                         isLocal,
        const tarch::la::Vector<Dimensions, double>& vertexX,
        const tarch::la::Vector<Dimensions, double>& vertexH,
        int                                          treeId,
        const std::string&                           trace
      );

      /**
       * Assign a particle to a sieve set
       *
       * To make this routine valid, the particle may not be assigned to a
       * vertex anymore, i.e. detachParticleFromVertex() has to be called
       * before.
       */
      void assignParticleToSieveSet(
        const std::string&                           particleName,
        const tarch::la::Vector<Dimensions, double>& particleX,
        const int                                    particleID,
        bool                                         isLocal,
        const tarch::la::Vector<Dimensions, double>& vertexH,
        int                                          treeId,
        const std::string&                           trace
      );

      /**
       * Find out particle name
       *
       * In some places, it is convenient to construct the particle name from
       * the type. However, some compilers only give you something similar to
       *
       * ~~~~~~~~~~~~~~~~~~~~
       * N10benchmarks6swift23noh10globaldata9HydroPartE
       * ~~~~~~~~~~~~~~~~~~~~
       *
       * which is really not what you want. This routine tries to reconstruct a
       * meaningful result.
       */
      template <typename Particle> std::string pruneTypeName() {
        std::string result    = typeid(Particle).name();
        std::size_t lastDigit = result.find_last_of("0123456789");
        if (lastDigit != std::string::npos) {
          result = result.substr(lastDigit + 1);
        }
        assertion2(result.size() > 1, typeid(Particle).name(), result);
        return result.size() > 1 ? result.substr(0, result.size() - 1) : result;
      }

      /**
       * Ensure that database is empty
       *
       * This one is usually called after eliminateExistingParticles() and will
       * leave the user only with particles which have "magically" disappeared.
       *
       * This routine uses exit(-1) if no assertions are enabled.
       */
      void ensureDatabaseIsEmpty();

    } // namespace assignmentchecks
  }   // namespace particles
} // namespace toolbox
