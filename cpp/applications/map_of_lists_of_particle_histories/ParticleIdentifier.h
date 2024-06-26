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

        struct ParticleIdentifier;
        struct ParticleSearchIdentifier;

        struct ParticleIdentifier {

          // don't make these functions virtual: I need to be able to pass
          // &ParticleSearchIdentifier and have it be treated like
          // &ParticleIdentifier in function arguments.

          std::string                           particleName;
          tarch::la::Vector<Dimensions, double> particleX;
          int                                   particleID;

          std::string toString() const;

          bool numericalEquals(const ParticleIdentifier& rhs) const;
          bool numericalSmaller(const ParticleIdentifier& rhs) const ;
          bool numericalSmallerWithTolerance(const ParticleIdentifier& rhs, const double tolerance) const;

          bool operator==(const ParticleIdentifier& rhs) const;
          bool operator!=(const ParticleIdentifier& rhs) const;
          bool operator<(const ParticleIdentifier& rhs) const;
          bool operator<(const ParticleSearchIdentifier& rhs) const;


          ParticleIdentifier(
            const std::string&                           particleName__,
            const tarch::la::Vector<Dimensions, double>& particleX__,
            const int                                    particleID__
          );
        };


        struct ParticleSearchIdentifier: ParticleIdentifier {

          static constexpr double shiftTolerance = 0.3;

          double positionTolerance;

          bool operator<(const ParticleIdentifier& rhs) const;

          ParticleSearchIdentifier(
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
/* bool operator==( */
/*   const toolbox::particles::assignmentchecks::internal::ParticleIdentifier& lhs, */
/*   const toolbox::particles::assignmentchecks::internal::ParticleIdentifier& rhs */
/* ); */
/* bool operator!=( */
/*   const toolbox::particles::assignmentchecks::internal::ParticleIdentifier& lhs, */
/*   const toolbox::particles::assignmentchecks::internal::ParticleIdentifier& rhs */
/* ); */
/* bool operator<( */
/*   const toolbox::particles::assignmentchecks::internal::ParticleIdentifier& lhs, */
/*   const toolbox::particles::assignmentchecks::internal::ParticleIdentifier& rhs */
/* ); */
/*  */
/* bool operator==( */
/*   const toolbox::particles::assignmentchecks::internal::ParticleSearchIdentifier& lhs, */
/*   const toolbox::particles::assignmentchecks::internal::ParticleSearchIdentifier& rhs */
/* ); */
/* bool operator!=( */
/*   const toolbox::particles::assignmentchecks::internal::ParticleSearchIdentifier& lhs, */
/*   const toolbox::particles::assignmentchecks::internal::ParticleSearchIdentifier& rhs */
/* ); */
/* bool operator<( */
/*   const toolbox::particles::assignmentchecks::internal::ParticleSearchIdentifier& lhs, */
/*   const toolbox::particles::assignmentchecks::internal::ParticleSearchIdentifier& rhs */
/* ); */
/*  */
/* bool operator<( */
/*   const toolbox::particles::assignmentchecks::internal::ParticleIdentifier& lhs, */
/*   const toolbox::particles::assignmentchecks::internal::ParticleSearchIdentifier& rhs */
/* ); */
/*  */
/* bool operator<( */
/*   const toolbox::particles::assignmentchecks::internal::ParticleSearchIdentifier& lhs, */
/*   const toolbox::particles::assignmentchecks::internal::ParticleIdentifier& rhs */
/* ); */
/*  */
/*  */


/* namespace { */
/*  */

/* bool operator==( */
/*   const toolbox::particles::assignmentchecks::internal::ParticleSearchIdentifier& lhs, */
/*   const toolbox::particles::assignmentchecks::internal::ParticleSearchIdentifier& rhs */
/* ) { */
/*   return lhs.numericalEquals(rhs); */
/* } */
/* bool operator!=( */
/*   const toolbox::particles::assignmentchecks::internal::ParticleSearchIdentifier& lhs, */
/*   const toolbox::particles::assignmentchecks::internal::ParticleSearchIdentifier& rhs */
/* ) { */
/*   return not lhs.numericalEquals(rhs); */
/* / *[> } */
/* const bool operator<( */
/*   const toolbox::particles::assignmentchecks::internal::ParticleSearchIdentifier& lhs, */
/*   const toolbox::particles::assignmentchecks::internal::ParticleSearchIdentifier& rhs){ */
/*   return lhs.numericalSmaller(rhs); */
/* } */
/* const bool operator<( */
/*   const toolbox::particles::assignmentchecks::internal::ParticleSearchIdentifier lhs, */
/*   const toolbox::particles::assignmentchecks::internal::ParticleSearchIdentifier rhs){ */
/*   return lhs.numericalSmaller(rhs); */
/* } */
/*  */
/*  */
/*  */
/*  */
/*  */
/* const bool operator<( */
/*   const toolbox::particles::assignmentchecks::internal::ParticleIdentifier& lhs, */
/*   const toolbox::particles::assignmentchecks::internal::ParticleSearchIdentifier& rhs){ */
/*   return lhs.numericalSmallerWithTolerance(rhs, rhs.positionTolerance); */
/* } */
/*  */
/*  */
/* const bool operator<( */
/*   const toolbox::particles::assignmentchecks::internal::ParticleSearchIdentifier& lhs, */
/*   const toolbox::particles::assignmentchecks::internal::ParticleIdentifier& rhs){ */
/*   return lhs.numericalSmallerWithTolerance(rhs, lhs.positionTolerance); */
/* } */
/*  */
/*  */
/* const bool operator<( */
/*   const toolbox::particles::assignmentchecks::internal::ParticleIdentifier lhs, */
/*   const toolbox::particles::assignmentchecks::internal::ParticleSearchIdentifier rhs){ */
/*   return lhs.numericalSmallerWithTolerance(rhs, rhs.positionTolerance); */
/* } */
/*  */
/*  */
/* const bool operator<( */
/*   const toolbox::particles::assignmentchecks::internal::ParticleSearchIdentifier lhs, */
/*   const toolbox::particles::assignmentchecks::internal::ParticleIdentifier rhs){ */
/*   return lhs.numericalSmallerWithTolerance(rhs, lhs.positionTolerance); */
/* } */
/*  */
// }
