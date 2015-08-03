<?php
namespace library;
use app\model\file;
use library\filter;

class utils{
    private static $seed = null;
    private static $uniqueRand = array();
    
    public static function log($msg,$file = "error_log.txt"){
        file::append($file,$msg."\n");   
    }
    public static function randomGenerator($seed, $unique = true, $min = 1, $max = 99999, $attempt = 100){
        if(self::$seed == null){
            mt_srand($seed);
            self::$seed = $seed;
        }
        if($unique){
            return self::getUniqueNumber($min, $max, $attempt);
        } else {
            return mt_rand($min,$max);
        }
    }
    private static function getUniqueNumber($min,$max,$attempt){
        $i = 0;
        while(true){
            $rand = mt_rand($min,$max);
            if(!in_array($rand,self::$uniqueRand)){
                self::$uniqueRand[] = $rand;
                return $rand;
            }
            $i++;
            if($i >= $attempt){
                throw new \Exception("Could not generate unique number");
            }
        }
    }
    
    public static function array_remove($array,$value){
        foreach ($array as $key => $element) {
            if ($element == $value) {
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