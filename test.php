<?php 
mysql_connect('localhost', 'upestech', 'vaticination');
mysql_select_db('vaticination');
$qry = mysql_query("SELECT * FROM data_details");
$data = "";
while($row = mysql_fetch_array($qry)) {
  $data .= $row['age'].",".$row['sex'].",".$row['cp'].",".$row['bp'].",".$row['sc'].",".$row['fs'].",".$row['re'].",".$row['mh'].",".$row['ig'].",".$row['st'].",".$row['stg'].",".$row['nv'].",".$row['th']."\n";
}


echo $data; exit();
?>