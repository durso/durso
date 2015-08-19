<?php
namespace library;
use app\model\file;


class utils{
    private static $seed = null;
    private static $uniqueRand = array();
    
    public static function log($msg,$file = "error_log.txt"){
        file::append($file,$msg."\n");   
    }
    public static function randomGenerator($min = 1, $max = 9999999){

        if(self::$seed == null){
            self::$seed = time();
            mt_srand(self::$seed);      
        }

        return mt_rand($min,$max);
    }
    
    public static function isUrl($url){
        return filter_var($url, FILTER_VALIDATE_URL);
    }

    
    public static function array_remove($array,$value){
        foreach ($array as $key => $element) {
            if ($element === $value) {
                unset($array[$key]);
            }
        }
        $array = array_values($array);
        return $array;
    }
    public static function runMethods($className){
        $class = new $className;
        $methods = get_class_methods($class);
        foreach($methods as $method){
            $class->$method();
        }
    }
    
}