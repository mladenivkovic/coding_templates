#include <iostream> // IO library


void switch_exam_score(int score){
  // Assume score is between 0-10

  switch (score) {
    case 10:
      std::cout << "Perfect score!" << std::endl;
      break;
    case 9:  // "fall throgh": execute everything until next break statement
    case 8: 
      std::cout << "Excellent!" << std::endl;
      break;
    case 7: 
    case 6:
      std::cout << "Well done!" << std::endl;
      break;
    case 5: case 4:
      std::cout << "You could do better next time" << std::endl;
      break;
    default: // If no other case was caught, go with the default
      std::cout << "You bring shame on your family." <<std::endl;
      break;
  }
}


int main(){

  // Basic switching

  for (int score = 10; score > 0; score--){
    std::cout << "Your score is " << score << " - ";
    switch_exam_score(score);
  }


  return 0;
}


