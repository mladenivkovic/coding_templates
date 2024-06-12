#include "MeshSweepData.h"

#include <string>

toolbox::particles::assignmentchecks::internal::MeshSweepData::MeshSweepData(
  const std::string& meshSweepName
):
  _meshSweepName(meshSweepName) {}


std::string toolbox::particles::assignmentchecks::internal::MeshSweepData::
  getName() const {
  return _meshSweepName;
}


