// This file is part of the Peano project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once

#include "tarch/la/Vector.h"

namespace toolbox {
  namespace particles {
    namespace assignmentchecks {
      namespace internal {

        /**
         * Event
         *
         * An event is an assocation or de-association of a particle and a
         * vertex.
         *
         * TODO: update documentation here.
         */
        struct Event {
          enum class Type {
            NotFound,
            AssignToSieveSet,
            AssignToVertex,
            Erase,
            DetachFromVertex,
            MoveWhileAssociatedToVertex
          };

          Type                                  type;
          bool                                  isLocal;
          tarch::la::Vector<Dimensions, double> vertexX;
          tarch::la::Vector<Dimensions, double> previousParticleX;
          tarch::la::Vector<Dimensions, double> vertexH;
          int                                   treeId;
          std::string                           trace;
          int  meshSweepIndex;

          /**
           * Construct an event which identifies a vertex assignment or the
           * removal from a vertex.
           */
          Event(
            Type                                         type_,
            bool                                         isLocal_,
            const tarch::la::Vector<Dimensions, double>& vertexX_,
            const tarch::la::Vector<Dimensions, double>& particleX,
            const tarch::la::Vector<Dimensions, double>& vertexH_,
            int                                          treeId_,
            const std::string&                           trace_
          );

          /**
           * Construct a move event
           *
           * This event should always be tied to an identifier in the global map
           * which uses the new position as key.
           */
          Event(
            Type                                         type_,
            const tarch::la::Vector<Dimensions, double>& vertexX_,
            const tarch::la::Vector<Dimensions, double>& previousParticleX_,
            const tarch::la::Vector<Dimensions, double>& vertexH_,
            int                                          treeId_,
            const std::string&                           trace_
          );

          /**
           * Construct an event that identifies the association to the sieve set
           * or an erase.
           */
          Event(
            Type               type_,
            bool               isLocal_,
            int                treeId_,
            const std::string& trace_
          );

          /**
           * Construct an invalid event. Is used as return value if no data is
           * found in the database.
           */
          Event(Type type_);

          std::string toString() const;
        };
      }
    }
  }
}
