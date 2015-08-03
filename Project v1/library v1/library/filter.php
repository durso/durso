<?php
/**
 * Description of filter
 *
 * @author durso
 */
namespace library;

class filter {
    const ALL = "([^/]+)";
    const ALNUM = "([\w-]+)"; 
    const ALPHA = "([a-zA-Z]+)"; 
    const INT = "([\d]+)";
    const STRING = "([\w-]+)";
    const NAME = "([\w\s'])";
    const ID = "#([\w-]+)";
    const PATH = "([\w-\/]+)";
    
    public static function validate($string, $pattern){
        return preg_match($pattern, $string);
    }

    public static function getRegex($pattern, $final = false){
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
            case "name":    $regex = self::NAME;
                            break;                
            default:        $regex = self::STRING;
                            break;               
        }
        if($final){
            return "/".$regex."/";
        }
        return $regex;    
    }
}
