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
         * Mesh sweep data.
         *
         * For now, only contains the name of the current sweep as a string.
         * Keeping this as a separate class in case we decide one day that
         * we need more associated data, e.g. current simulation time,
         * timings, or whatnot.
         *
         */

        class MeshSweepData{

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

