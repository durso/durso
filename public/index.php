<?php



ob_start();
error_reporting(E_ALL);
ini_set('display_errors', 1);

chdir(dirname(__DIR__));

use app\loader;
use app\init;
use app\bootstrap;

require_once("config".DIRECTORY_SEPARATOR."config.php");
require_once(APP_PATH.DS."loader.php");

$loader = new loader;
$loader->register();
$init = new init;
$init->exceptionHandler();
$boot = new bootstrap($_GET); 
$boot->createController();

?>