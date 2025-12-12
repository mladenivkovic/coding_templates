#!/bin/bash


#=====================================
# Prompt the user for confirmation.
#=====================================


while true; do
	read -p "Do you wish to continue? (y/n) " yn
    case $yn in
      [Yy]* ) echo "Program continues."; break;;
      [Nn]* ) echo "exiting."; exit;;
      * ) echo "Please answer yes or no.";;
    esac
done

echo "Program finished properly."
