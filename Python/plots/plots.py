#!/usr/bin/python

# A script to demonstrate various plotting options.
# The resulting figure will be saved as plotsoutput.png

#import matplotlib as mpl
#mpl.use('Agg') #dont try to show anything/use Xorg/Xserver unless I specifically ask you to.
import matplotlib.pyplot as plt
import numpy as np



fig = plt.figure(facecolor='white', figsize=(10,12))
# mit figure() wird bei jedem Setzen eine neue Graphik aufgemacht
# other options: facecolor='somecolor', figsize=(widthininches,heightininches), dpi=somenint





# add supertitle for the whole picture
fig.suptitle("Suptitle")


#some bullshit we want to plot

x=np.linspace(0,4000,100)
f=np.exp(x/1000)
g=np.exp(-x/1000)



# add subplots
rows=3
columns=2

ax1=plt.subplot(rows, columns, 1)



#Plot 1
ax1.plot(x,f,'r',label='linie1')
ax1.plot(x,np.log10(f),'b--',label='linie2')
ax1.set_xlabel('x-Achse')
ax1.set_ylabel('y-Achse')
ax1.set_title('Titel plot')
ax1.legend(loc=0)
ax1.grid()
ax1.set_xticks(range(0,4001,1000)) #x-Achsen-Ticks setzen







#Plot 2
ax2 = plt.subplot(rows,columns,2)
ax2.semilogy(x,f,'g',label='linie3')
ax2.set_xlabel('x-Achse')
ax2.set_ylabel('y-Achse')
ax2.set_title('Titel semilogy')
#ax2.set_xticks(range(0,4001,1000)) #x-Achsen-Ticks setzen
ax2.grid()
ax2.legend(loc=0)





#Plot3 Polynom mit Nullstellen und Grid
ax3 = plt.subplot(rows,columns,3)
a = np.linspace(-5,3.5,100)
C=[1, 3, -13, -15]
fx = np.polyval(C,a)

ax3.plot(a,fx,'b', label='Polynom')
ax3.plot(np.roots(C), np.zeros(len(np.roots(C))), 'ro', label='Nullstellen') #Nullstellen
ax3.axis([-5.5,3.5,-30,30])
ax3.grid() #Gitter
ax3.set_title('Polynom 3. Ordnung')
ax3.set_xlabel('Abszisse')
ax3.set_ylabel('Ordinate')
ax3.legend(loc=0)


#Plot4 Halblogarithmischer Plot
ax4 = plt.subplot(rows,columns,4)
ax4.semilogy(x,g,'y',label='linie6')
ax4.set_xticks(range(0,4001,1000)) #x-Achsen-Ticks setzen
ax4.set_xlabel('x-Achse')
ax4.set_ylabel('y-Achse')
ax4.set_title('Titel')
ax4.grid(which='both')
ax4.legend(loc=0)



#Plot5 - Fehlerplot 1
ax5=plt.subplot(rows,columns,5)
ax5.errorbar(range(0,5,1),[10,9,8,7,6],[0.6,0.7,0.5,0.8,0.7],[0.1,0.2,0.1,0.3,0.2],'r.', label='name1')
ax5.set_xlabel('x-Achse')
ax5.set_ylabel('y-Achse')
ax5.set_title('Titel')
ax5.grid()
ax5.legend(loc=0)



#Plot6 - Fehlerplot 2
ax6=plt.subplot(rows,columns,6)
ax6.errorbar(range(0,5,1),[10,9,8,7,6],[[0.2,0.2,0.3,0.2,0.1],[0.3,0.4,0.5,0.4,0.2]], xerr=0.2, fmt='g.', label='name2')
ax6.set_xlabel('x-Achse')
ax6.set_ylabel('y-Achse')
ax6.set_title('Titel')
ax6.grid()
ax6.legend(loc=0)





plt.tight_layout() #let python create a nice layout
print "Figure created"
    
    
# saving figure
outputfilename='plots.png'
print "saving figure as "+outputfilename
plt.savefig(outputfilename, format='png', transparent=False, dpi=100)

print "done"
plt.show()
plt.close()



