class Rectangle {
        int width, height;
    public:
        //constructurs:
        Rectangle(int,int);
        Rectangle ();

        //methods:
        void set_values (int,int);
        int area() {return width*height;}
};

