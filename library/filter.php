<?php
/**
 * Description of filter
 *
 * @author durso
 */
namespace library;

class filter {
    const ALL = "([^/]+)";
    const ALNUM = "([\d\w-]+)"; 
    const ALPHA = "([a-zA-Z]+)"; 
    const INT = "([\d]+)";
    const STRING = "([\w-]+)";
    const ID = "(#[\d\w-]+)";
    const PATH = "([\d\w-\/]+)";
    
    public static function validate($string, $pattern){
        return preg_match($pattern, $string);
    }

    public static function getRegex($pattern){
        $regex;
        switch (strtolower($pattern)) { 
            case "int":     $regex = self::INT;                         
                            break;                    
            case "alpha":   $regex = self::ALPHA;
                            break;                       
            case "alnum":   $regex = self::ALNUM;
                            break;  
            case "id":      $regex = self::ID;
                            break;
            case "path":    $regex = self::PATH;
                            break;            
            default:        $regex = self::STRING;
                            break;               
        }
        return $regex;    
    }
}
