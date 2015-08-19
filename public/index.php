<?php




ob_start();
error_reporting(E_ALL & ~E_STRICT);
ini_set('display_errors', 1);
assert_options(ASSERT_BAIL, 1);
session_start();





chdir(dirname(__DIR__));

use app\loader;
use app\init;


require_once("config".DIRECTORY_SEPARATOR."config.php");
require_once(APP_PATH.DS."loader.php");

$loader = new loader;
$loader->register();
$init = new init;
$init->run();


?>