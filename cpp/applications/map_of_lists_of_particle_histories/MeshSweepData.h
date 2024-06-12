// This file is part of the Peano project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once

#include <map>
#include "ParticleIdentifier.h"
#include "Event.h"

namespace toolbox {
  namespace particles {
    namespace assignmentchecks {
      namespace internal {


        /**
         * Mesh sweep data set
         *
         * TODO: Docu wrong
         * All the traces bookkept for one single mesh sweep. This is a plain
         * extension of the std::map which simply adds a unique name on top.
         * So all the original semantics of a map are preserved.
         *
         * Please see ParticleIdentifier for an in-depth discussion of the
         * comparison key to be used here.
         */

        using ParticleEvents = std::vector<Event>;

        class MeshSweepData:
          public std::map<ParticleIdentifier, ParticleEvents> {
        public:
          MeshSweepData(const std::string& meshSweepName);

          std::string getName() const;

        private:
          const std::string _meshSweepName;
        };

      }
    }
  }
}

