#include <iostream>
#include <string>
using namespace std;


// define structure
struct movies_t {
  string title;
  int year;
} movie;

void printmovie (movies_t movie);

int main ()
{
    string mystr;

    movies_t firstmovie;
    firstmovie.title="Blade Runner";
    firstmovie.year=1982;
    printmovie(firstmovie);

    movies_t secondmovie;
    secondmovie.title="The Matrix";
    secondmovie.year=1999;
    printmovie(secondmovie);

}

void printmovie (movies_t movie)
{
  cout << movie.title;
  cout << "\t (" << movie.year << ")\n";
}
