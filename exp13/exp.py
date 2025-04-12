import matplotlib . pyplot as plt
# Initialize data
x = [10 , 20 , 30 , 40]
y = [20 , 25 , 35 , 55]
# Create plot
plt . plot (x , y )
plt . xlabel ( ’X - axis ’)
plt . ylabel ( ’Y - axis ’)
plt . title ( ’ Line Plot Example ’)
plt . show ()

import matplotlib . pyplot as plt
# Age data for 100 individuals
ages = [
1 ,1 ,2 ,3 ,3 ,5 ,7 ,8 ,9 ,10 ,10 ,11 ,11 ,13 ,13 ,15 ,16 ,17 ,18 ,18 ,
18 ,19 ,20 ,21 ,21 ,23 ,24 ,24 ,25 ,25 ,25 ,25 ,26 ,26 ,27 ,27 ,27 ,27 ,27 ,
29 ,30 ,30 ,31 ,33 ,34 ,34 ,35 ,36 ,36 ,37 ,37 ,38 ,38 ,39 ,40 ,41 ,41 ,42 ,
43 ,44 ,45 ,45 ,46 ,47 ,48 ,48 ,49 ,50 ,51 ,52 ,53 ,54 ,55 ,56 ,57 ,58 ,60 ,
61 ,63 ,64 ,65 ,66 ,68 ,70 ,71 ,72 ,74 ,75 ,77 ,81 ,83 ,84 ,87 ,89 ,90 ,91
]
plt . hist ( ages , bins =10 , edgecolor = ’ black ’)
plt . xlabel ( ’ Age Groups ’)
plt . ylabel ( ’ Frequency ’)
plt . title ( ’ Age Distribution Histogram ’)
plt . show ()


