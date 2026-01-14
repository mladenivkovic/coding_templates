public class OOPBasics {
  public static void main(String[] args) {


    Cow cow1 = new Cow("Bella", 450.5);

    // This doesn't work if you don't provide a separate constructor for it
    Cow cow2 = new Cow();

    cow1.displayInfo();
    cow2.displayInfo();

    Cow cow3 = cow1;
    cow3.setName("Name changed");
    cow1.displayInfo();
    cow3.displayInfo();

  }
}
