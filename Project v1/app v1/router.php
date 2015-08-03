<?php
/**
 * Description of router
 *
 * @author durso
 */
namespace app;
use library\filter, app\request;

class router{
    
    /** * @var array The compiled routes */ 
    protected static $routes = array();
    protected static $route;
    protected static $hash = array();
    public static $match = false;
    

    /**  
     * Add a new route * * 
     * @param string $route The route pattern 
     */ 
    public static function addRoute($route, $options = array()) {  
        if(!array_key_exists("controller", $options)){
            throw new \Exception("Please provide the controller option in the route");
        }
        self::$routes[] = array("pattern" => self::parseRoute($route),"options" => $options); 
    }
    protected static function parseRoute($route){
        $parts = explode("/",$route);
        array_shift($parts);
        $pattern = self::getPattern($parts);
        return $pattern;
    }
    protected static function getPattern($parts){
        $pattern = "/^";
        $i = 0;
        foreach($parts as $part){
            $pattern .= "\/";
            if(strpos($part,':') !== false){
                if(strpos($part,0,1) === ":"){
                    $type = substr($part,1,strlen($part));
                    $pattern .= filter::getRegex($type);
                } else {
                    $array = explode(":",$part);
                    $pattern .= "(?<".$array[0].">".filter::getRegex($array[1]).")";
                }
            } else {
                $pattern .= $part;
            }
            $i++;
        }
        $pattern .= "/";
        return $pattern;
    }
    public static function checkMatch($path){
        foreach(self::$routes as $route){
            if(preg_match($route["pattern"],$path,$matches)){
                self::$route = $route["options"];
                foreach($matches as $key => $value){
                    if(is_int($key)){
                        unset($matches[$key]);
                    }
                }
                self::$match = $matches;
                return true;
            }
        }
        return false;
    }
    public static function hasMatch(){
        return is_array(self::$match);
    }
    public static function getMatch(){
        return self::$match;
    }
    public static function getController(){
        return self::$route["controller"];
    }
    public static function getAction(){
        if(isset(self::$route["action"]) && !empty(self::$route["action"])){
            return self::$route["action"];
        }
        return "index";
    }
    
    
    
    
   
}
