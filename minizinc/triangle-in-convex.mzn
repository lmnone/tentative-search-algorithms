QuadrFloat=true;


% config vars
int: sign_target_ac_dx;
int: sign_target_ac_dy;
int: sign_slopes;
int: sign_m;

float: eps = 1e-3;
float: online_eps = 2.0;

int: a_len;
int: b_len;
int: c_len;

% A define 
array [1..a_len] of 0..500: a_x;

array [1..a_len] of 0..500: a_y;

array [1..a_len] of float: a_slopes;


% B define
array [1..b_len] of 0..500: b_x;

array [1..b_len] of 0..500: b_y;

array [1..b_len] of float: b_slopes;


% C define
array [1..c_len] of 0..500: c_x;

array [1..c_len] of 0..500: c_y;

array [1..c_len] of float: c_slopes;

var 1..a_len: a;

var 1..b_len: b;

var 1..b_len: c;

var -1e-4..1e4: m_ac;
var float: m;

var 0.0..500.0: a_x_refine;
var 0.0..500.0: a_y_refine;

var 0.0..500.0: b_x_refine;
var 0.0..500.0: b_y_refine;

var 0.0..500.0: c_x_refine;
var 0.0..500.0: c_y_refine;


function var int: get_value_from_bounded_list(array [int] of int: P, var int: i, var int: n) = 
  P[min(n,i)];
    

predicate coordinate_in_interval(array [int] of int: P, var int: i, var int: n, var float: p) =
  let {
    var int : p1 = get_value_from_bounded_list(P, i, n);
    var int : p2 = get_value_from_bounded_list(P, i+1, n);
    

  }    in ( (p1 >= p2 ->  p >= int2float(p2) /\ p <= int2float(p1)) /\ (p1 < p2 ->  p <= int2float(p2) /\ p >= int2float(p1)) );


predicate coordinate_on_line(array [int] of int: P_x, array [int] of int: P_y, var int: i, var int: n, var float: p_x, var float: p_y) =
  let {
    var int : p1_x = get_value_from_bounded_list(P_x, i, n);
    var int : p2_x = get_value_from_bounded_list(P_x, i+1, n);
    
    var int : p1_y = get_value_from_bounded_list(P_y, i, n);
    var int : p2_y = get_value_from_bounded_list(P_y, i+1, n);
    
    var float: a1 = int2float(p1_y - p2_y);
    var float: a2 = int2float(p2_x - p1_x);
    var float: b =  int2float(p2_x * p1_y - p1_x * p2_y);
    
    

  }    in ( abs(a1 * p_x + a2 * p_y - b) <= online_eps );


constraint coordinate_in_interval(a_x, a, a_len, a_x_refine) = 1;
constraint coordinate_in_interval(a_y, a, a_len, a_y_refine) = 1;

constraint coordinate_on_line(a_x, a_y, a, a_len, a_x_refine, a_y_refine) = 1;


constraint coordinate_in_interval(b_x, b, b_len, b_x_refine) = 1;
constraint coordinate_in_interval(b_y, b, b_len, b_y_refine) = 1;

constraint coordinate_on_line(b_x, b_y, b, b_len, b_x_refine, b_y_refine) = 1;

constraint coordinate_in_interval(c_x, c, c_len, c_x_refine) = 1;
constraint coordinate_in_interval(c_y, c, c_len, c_y_refine) = 1;

constraint coordinate_on_line(c_x, c_y, c, c_len, c_x_refine, c_y_refine) = 1;

var float: target_ac_dx = int2float(sign_target_ac_dx) * (b_x_refine - a_x_refine);
var float: target_ac_dy = int2float(sign_target_ac_dy) * (c_y_refine - b_y_refine);

constraint abs(target_ac_dx) >= eps;

% target_ac_dx, target_ac_dy should be both positive
constraint (m_ac * target_ac_dx >= target_ac_dy) ;



constraint abs(a_y_refine - b_y_refine) <= online_eps;
constraint abs(b_x_refine - c_x_refine) <= online_eps;

% slopes signs should be -1,+1,-1
var float: m_a = int2float(sign_slopes) * a_slopes[a];
var float: m_b = int2float(sign_slopes) * b_slopes[b];
var float: m_c = int2float(sign_slopes) * c_slopes[c];

% Apparently this is redundant.
%constraint -m_a >= m_ac;
%constraint m_ac >= -m_c;
%constraint -m_c > 0;

m = int2float(sign_m) * m_a * (m_c - m_b) / ( m_a - m_b);


constraint m_ac >= m;

solve minimize m_ac - m + (m_ac * target_ac_dx - target_ac_dy);
output [
  "target_ac_dx:", show(target_ac_dx), ", target_ac_dy:", show(target_ac_dy), ",m_ac:", show(m_ac), ",m:", show(m), ",m_a:", show(m_a), ",m_c:", show(m_c)

]




