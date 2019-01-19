<html>
<head>
  
	<title>Vaticination - User Symptoms Portal</title>
  
  <style type="text/css">
    body{
        text-align:center;
		background-image: url('http://vaticination.ga/heart.jpg');
  
    }
    

	
	
    </style>

  
</head>

<body>
<center>
    <h1>Vaticination : Dynamic User Symptoms Portal</h1>
  

<?php
session_start();
$conn = mysqli_connect(null,"upestech","vaticination","vaticination");
$result = mysqli_query($conn,"SELECT * FROM account_holder_details WHERE user_id='".$_SESSION['username']."'"); 


echo '<table class="text" border=1px>';  
echo '<th>User ID</th><th>First Name</th><th>Last Name</th><th>Address</th><th>City</th><th>State</th><th>Pincode</th><th>Phone</th>'; 

 while($data = mysqli_fetch_array($result))
{

 
echo'<tr>';
echo '<td>'.$data['user_id'].'</td><td>'.$data['first_name'].'</td><td>'.$data['last_name'].'</td><td>'.$data['address'].'</td><td>'.$data['city'].'</td><td>'.$data['state'].'</td><td>'.$data['pincode'].'</td><td>'.$data['phone'].'</td>'; 
echo'</tr>';
 
}

echo '</table>. </br><br/><br/>';

$new = mysqli_query($conn,"SELECT * FROM data_details WHERE user_id='".$_SESSION['username']."'"); 


echo '<table class="text" border=1px>';  
echo '<th>Age</th><th>Sex</th><th>Chest pain type</th><th>Blood_pressure</th><th>Serum cholestoral</th><th>Fasting blood sugar</th><th>Resting electrocardiographic</th><th>Max_heart_rate</th><th>induced_angina</th><th>ST depression</th><th>ST segment</th><th>Vessel</th><th>Thal</th>'; 

 while($data = mysqli_fetch_array($new))
{

 
echo'<tr>';
echo '<td>'.$data['age'].'</td><td>'.$data['sex'].'</td><td>'.$data['cp'].'</td><td>'.$data['bp'].'</td><td>'.$data['sc'].'</td><td>'.$data['fs'].'</td><td>'.$data['re'].'</td><td>'.$data['mh'].'</td><td>'.$data['ig'].'</td><td>'.$data['st'].'</td><td>'.$data['stg'].'</td><td>'.$data['nv'].'</td><td>'.$data['th'].'</td>'; 
echo'</tr>';
 
}

echo '</table>'; 




?> 
</br>
<b><p>Diagnosis value will be shown using machine learning based classifiers, predict below!</p></b></br>
<input type="button" onclick="location.href='https://vaticination.ga:8888';" value="Jupyter Server : Model Predict Outputs" />

  </center>

	
		



</body>

</html>
