// Basically a minimal class reproducing whatever Peano4 has going on
// so I can play around with it and get things to work

#pragma once

#include <list>
#include <string>
#include <vector>

/**
 * Dummy hydro particle
 */
struct hydroPart {
public:
  hydroPart() {}
  hydroPart(std::vector<double> __X, int __PartID) {
    _X = __X;
    _PartID = __PartID;
  }

  std::vector<double> getX(void) { return _X; }

  void setX(std::vector<double> X) { _X = X; }

  int getPartID(void) { return _PartID; }

  void setPartID(int PartID) { _PartID = PartID; }

  void sayHello() { std::cout << "Hello from particle " << _PartID << "\n"; }

private:
  std::vector<double> _X;
  int _PartID;
};

/**
 * Base class for particle sets.
 */
template <typename T> class ParticleSet {
public:
  typedef T DoFType;
  typedef std::list<T *> ParticleList;

  // private:
};

/**
 * Specific class for hydro particle sets.
 */
class hydroPartSet : ParticleSet<hydroPart> {
public:
  typedef ParticleSet<hydroPart> Base;
  typedef std::vector<hydroPart *> Container;

  /**
   * Expose C++ standard interface
   */
  typedef Container::value_type value_type;
  typedef Container::iterator iterator;
  typedef Container::const_iterator const_iterator;

  hydroPartSet() = default;

  void clear() { _container.clear(); }

  void deleteParticles() {
    for (auto *p : _container) {
      delete p;
    }
    _container.clear();
  }

  Container::iterator deleteParticle(const Container::iterator &particle) {
    delete *particle;
    return _container.erase(particle);
  }

  int size() const { return _container.size(); }

  void addParticle(DoFType *particle) { _container.push_back(particle); }

  Container::iterator begin() { return _container.begin(); }
  Container::iterator end() { return _container.end(); }
  Container::const_iterator begin() const { return _container.begin(); }
  Container::const_iterator end() const { return _container.end(); }

  /**
   * Just print out whatever particles you contain
   */
  void sanityCheck(std::string message) {
    std::cout << message << "\n";
    for (auto p : _container)
      std::cout << "(" << p->getPartID() << "/" << *(p->getX().begin())
                << "), ";
    std::cout << std::endl;
  }

private:
  Container _container;
};

/**
 * Generate a dummy hydroPartSet with N particles.
 */
hydroPartSet generateDummyPartSet(int N) {

  double dx = 1. / N;

  hydroPartSet newPartset;

  for (int i = 0; i < N; i++) {
    std::vector<double> x = {(i + 0.5) * dx};
    hydroPart *p = new hydroPart(x, i + 1);
    newPartset.addParticle(p);
  }

  return newPartset;
}

/**
 * Dummy function that takes particle as argument and returns
 * sometimes true, sometimes false. Just for playing around,
 * no deeper sense to it.
 */
bool workOnPart(hydroPart &particle) { return (particle.getPartID() % 2 == 0); }
