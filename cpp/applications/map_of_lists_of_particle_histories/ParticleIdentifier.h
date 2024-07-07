// This file is part of the Peano project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org

#pragma once

#include "tarch/la/Vector.h"

namespace toolbox {
  namespace particles {
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
