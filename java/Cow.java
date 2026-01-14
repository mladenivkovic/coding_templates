class Cow {

  private String name;
  private double weight;


  // constructor
  public Cow(String name, double weight) {
    this.name = name;
    this.weight = weight;
  }

  public Cow() {
    this.name = "None";
    this.weight = 0.;
  }

  public void setName(String name){
    this.name = name;
  }

  // Method to display cow information
  public void displayInfo() {
    System.out.println("Cow Name: " + name + ", Weight: " + weight + " kg");
  }


}
