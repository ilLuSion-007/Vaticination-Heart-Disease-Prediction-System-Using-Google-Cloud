<?php
session_start();    
if(isset($_POST['username'])) { 
   $_SESSION['username'] = $_POST['username'];
}
$message="";
if(count($_POST)>0) {
	$conn = mysqli_connect(null,"upestech","vaticination","vaticination");
	$result = mysqli_query($conn,"SELECT * FROM account_holder_details WHERE user_id='" . $_POST["username"] . "' and password = '". $_POST["password"]."'");
	$count  = mysqli_num_rows($result);
	if($count==0) {
		$message = "Invalid Username or Password!";
	} else {
		header('Location: http://www.vaticination.ga/profile.php'); 
	}
}
?>
<html>
<head>
	<title>Vaticination - Heart Disease Prediction System</title>

	<!-- Google Fonts -->
	<link href='https://fonts.googleapis.com/css?family=Roboto+Slab:400,100,300,700|Lato:400,100,300,700,900' rel='stylesheet' type='text/css'>

	
	
	<link rel="stylesheet" href="style.css">
</head>
<body>
	<div class="container">
		<div class="top">
			<h1 id="title" class="hidden"><span id="logo">Vaticination - Heart Disease Prediction System</span></span></h1>
		</div>
		<div class="login-box animated fadeInUp">
			<div class="box-header">
				<h2>Log In</h2>
			</div><form name="quiz" method="post" action="" >
			<label for="username">Username</label>
			<br/>
			<input type="text" name="username" placeholder="Enter user-id
" id="username">
			<br/>
			<label for="password">Password</label>
			<br/>
			<input type="password" name="password" placeholder="Enter password" id="password">
			<br/>
			<button type="submit">Sign In</button>
			<br/></br>
			<div class="message"><?php if($message!="") { echo $message; } ?></br>
                        <b><p>Registration Via Client-Side Executable</p></b>
                        <input type="button" onclick="location.href='http://vaticination.ga/Vaticination_x64_executable.exe';" value="Registeration" /></div>       
		
		</div>
	</div>
</body>
</html>
