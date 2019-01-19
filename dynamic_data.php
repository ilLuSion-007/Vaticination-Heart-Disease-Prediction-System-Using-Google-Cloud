<?php
$con=mysqli_connect(null,"upestech","vaticination","vaticination");

if (mysqli_connect_errno())
  {
  echo "Failed to connect to MySQL: " . mysqli_connect_error();
  }

$sql="SELECT * FROM data_details";

if ($result=mysqli_query($con,$sql))
  {
  
  while ($row=mysqli_fetch_row($result))
    {
printf("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",$row[1],$row[2],$row[3],$row[4],$row[5],$row[6],$row[7],$row[8],$row[9],$row[10],$row[11],$row[12],$row[13]);
    }

  mysqli_free_result($result);
}

mysqli_close($con);
?>
