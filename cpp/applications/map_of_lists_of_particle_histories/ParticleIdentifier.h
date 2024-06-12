// This file is part of the Peano project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once

#include "tarch/la/Vector.h"

namespace toolbox {
  namespace particles {
    /**
     * TODO: Move to Database or TracingAPI.h
     *
     * @namespace toolbox::particles::assignmentchecks Correctness checks of
     * particle-mesh assignment
     *
     * This API is inspired by SWIFT's swift2::dependencychecks as introduced
     * by Mladen: We keep some history of particle-mesh associations, trace new
     * mesh assignments and validate them against the recorded history. All the
     * tracing degenerates to empty routines if we work in release mode, so
     * there should not be any overhead.
     *
     * Most toolbox routines do automatically report against this API. However,
     * to make the checks work, you have to report particle movements manually.
     *
     * Furthermore, you might want to invoke
     * toolbox::particles::assignmentchecks::startMeshSweep() prior to your
     * traversals to ensure that you get a nice break-down of all mesh
     * transitions.
     *
     * ## Enable assignment checks
     *
     * By default, the assignment checks are enabled as soon as PeanoDebug is
     * set to a value greater than 0. However, you can explicitly disable it
     * by adding
     *
     * -DnoAssignmentChecks
     *
     * to your compile flags. This is important as assignment checks are very
     * expensive. The checks also might yield invalid errors once you run them
     * over multiple ranks, i.e with -DParallel.
     */
    namespace assignmentchecks {
      namespace internal {

        struct ParticleIdentifier {

          static double Precision;

          friend class Database;

          std::string                           particleName;
          tarch::la::Vector<Dimensions, double> particleX;
          int                                   particleID;
          double                                positionTolerance;

          bool numericalEquals(const ParticleIdentifier& rhs) const;
          bool operator<(const ParticleIdentifier& rhs) const;

          std::string toString() const;

        private:

          /**
           * Never create a particle identifier manually
           *
           * Instead, use Database::createParticleIdentifier() to construct
           * one. This routine takes care of floating point in accuracies.
           */
          ParticleIdentifier(
            const std::string&                           particleName__,
            const tarch::la::Vector<Dimensions, double>& particleX__,
            const int                                    particleID__,
            const double                                 positionTolerance__
          );
        };
      }
    }
  }
}


/**
 * Comparison operator
 *
 * Some older variants of C++ require an external operator (not embedded
 * into class or namespace) to compare two objects. We have seen issues
 * with Intel 2023, e.g., while 2024 was absolutely fine. Delegates to
 * member function.
 */
bool operator==(
  const toolbox::particles::assignmentchecks::internal::ParticleIdentifier& lhs,
  const toolbox::particles::assignmentchecks::internal::ParticleIdentifier& rhs
);
bool operator!=(
  const toolbox::particles::assignmentchecks::internal::ParticleIdentifier& lhs,
  const toolbox::particles::assignmentchecks::internal::ParticleIdentifier& rhs
);

