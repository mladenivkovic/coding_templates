#include <fstream>  // file stream library
#include <iostream> // IO library
#include <string>   // strings

int main() {

  // Writing files
  // ==========================

  std::cout << "Writing to file 'example_output_to_file.txt'" << std::endl;

  // open file
  std::ofstream myOutputFile("example_output_to_file.txt");

  // check whether it's open
  if (myOutputFile.is_open()) {
    // write stuff
    myOutputFile << "This is a line.\n";
    myOutputFile << "This is another line.\n";

    // close again
    myOutputFile.close();
  }

  // myOutputFile << "This is another line.\n"; // this does nothing... throws
  // no error, doesn't write into file.

  // Reading files
  // ==========================

  std::cout << "Reading from file 'example_output_to_file.txt'" << std::endl;

  // Now read the file you wrote before back in again.

  std::string line; // buffer to store read in line again

  // open file
  std::ifstream myInputFile("example_output_to_file.txt");
  // check is open correctly
  if (myInputFile.is_open()) {
    // read line by line
    while (std::getline(myInputFile, line)) {
      std::cout << ">>" << line << '\n';
    }
    myInputFile.close();
  } else
    std::cout << "Unable to open file";

  // More generic way of opening file streams:
  // =========================================

  // syntax: stream.open( filename, mode )
  // stream is an instance of
  //   ofstream   for output
  //   ifstream   for input
  //   fstream    either input or output. must specify mode  ios::in | ios::out
  //   (see below)
  //
  // choices for 'mode':
  //   ios::in      Open for input operations.
  //   ios::out     Open for output operations.
  //   ios::binary  Open in binary mode.
  //   ios::ate     Set the initial position at the end of the file.  If this
  //   flag is not set, the initial position is the beginning of the file.
  //   ios::app     All output operations are performed at the end of the file,
  //   appending the content to the current content of the file. ios::trunc   If
  //   the file is opened for output operations and it already existed, its
  //   previous content is deleted and replaced by the new one.
  //
  // All these flags can be combined using the bitwise operator OR (|). e.g.
  //   std::open("whatever.txt", ios::in | ios::binary | ios::app )

  std::fstream myStream;
  myStream.open("example_general_open.txt", std::ios::out | std::ios::binary);
  if (myStream.is_open()) {
    myStream << "hello there\n";
    myStream << "General Kenobi\n";
    myStream.close();
  } else
    std::cout << "Couldn't open fstream myStream\n";

  myStream.open("example_general_open.txt", std::ios::in | std::ios::binary);
  std::string binaryline;
  if (myStream.is_open()) {
    while (std::getline(myStream, binaryline)) {
      std::cout << "'" << binaryline << "'";
    }
    myStream.close();
  } else
    std::cout << "Couldn't open fstream myStream\n";
  std::cout << "\n";

  return 0;
}
