
enum E = { Greater, Less }; % function eq
enum A = { Inc, Dec }; % function type

array [0..1] of var 0.0..1.0: _X;
array [0..1] of var 0.0..1.0: _Y;
array [0..1] of var 0.0..1.0: _Z;

array[0..1]  of var bool: x;
array[0..1]  of var bool: y;
array[0..1]  of var bool: z;

array[1..2] of var E: functions; % = [Less,Less];
array[1..2] of var A: composition; % = [Inc,Dec];


predicate monotone_inc(array[0..1] of var float:X, array[0..1] of var float:Y) =

          (X[0] > X[1] -> Y[0] > Y[1]) /\ (X[0] < X[1] -> Y[0] < Y[1]) /\ (X[0] = X[1] -> Y[0] = Y[1]);

predicate monotone_dec(array[0..1] of var float:X, array[0..1] of var float:Y) =

          (X[0] > X[1] -> Y[0] < Y[1]) /\ (X[0] < X[1] -> Y[0] > Y[1]) /\ (X[0] = X[1] -> Y[0] = Y[1]);
          

set of int: NUM = 0..1;
predicate greater_4_inc(array[0..1] of var float:X, array[0..1] of var float:Y) = 
     forall(i in NUM)(
     ( X[i] >= 0.5 -> Y[i] > 0.5) /\ (Y[i] <= 0.5 -> X[i] < 0.5)
     );
     
predicate less_4_inc(array[0..1] of var float:X, array[0..1] of var float:Y) = 
     forall(i in NUM)(
     ( X[i] <= 0.5 -> Y[i] < 0.5) /\ (Y[i] >= 0.5 -> X[i] > 0.5)
     );     
     
predicate greater_4_dec(array[0..1] of var float:X, array[0..1] of var float:Y) = 
     forall(i in NUM)(
     ( X[i] <= 0.5 -> Y[i] > 0.5) /\ (Y[i] <= 0.5 -> X[i] > 0.5)
     ); 

predicate less_4_dec(array[0..1] of var float:X, array[0..1] of var float:Y) = 
     forall(i in NUM)(
     ( X[i] >= 0.5 -> Y[i] < 0.5) /\ (Y[i] >= 0.5 -> X[i] < 0.5)
     ); 




predicate all_up(array[0..1] of var float:X) = 
     forall(i in NUM)(
     ( X[i] >= 0.5)
     );
predicate all_lo(array[0..1] of var float:X) = 
     forall(i in NUM)(
     ( X[i] <= 0.5)
     );


constraint all_up(_X) = x[1];
constraint all_lo(_X) = x[0];
constraint all_up(_Y) = y[1];
constraint all_lo(_Y) = y[0];
constraint all_up(_Z) = z[1];
constraint all_lo(_Z) = z[0];



predicate greater_4(array[0..1] of var float:X, array[0..1] of var float:Y, int: j) =
     if composition[j] = Inc then greater_4_inc(X,Y) else greater_4_dec(X,Y)
     endif;
     
predicate less_4(array[0..1] of var float:X, array[0..1] of var float:Y, int: j) =
     if composition[j] = Inc then less_4_inc(X,Y) else less_4_dec(X,Y)
     endif;

constraint if functions[1] = Greater then greater_4(_X,_Y, 1)=1 else less_4(_X,_Y, 1)=1 endif;
constraint if functions[2] = Greater then greater_4(_Y,_Z, 2)=1 else less_4(_Y,_Z, 2)=1 endif;

constraint if composition[1] = Inc then monotone_inc(_X,_Y)=1 else monotone_dec(_X,_Y)=1 endif;
constraint if composition[2] = Inc then monotone_inc(_Y,_Z)=1 else monotone_dec(_Y,_Z)=1 endif;

constraint (max(_Z) - min(_Z)) >= 0.5 -> (max(_X) - min(_X)) > 0.5 \/ (max(_Y) - min(_Y)) > 0.5;
constraint max(_Z) - min(_Z) >= 0.5;


constraint sum(x) < 2;
constraint sum(y) < 2;
%constraint sum(z) < 2;

constraint z=x;

solve maximize sum(x) + sum(y);
output [
  show(functions),",",show(_X),",",show(_Y),",",show(_Z),",",show(x),",",show(y)
]




