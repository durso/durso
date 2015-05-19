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
        foreach($parts as $part){
            $pattern .= "\/";
            if(substr($part,0,1) === ":"){
                $type = substr($part,1,strlen($part));
                $pattern .= filter::getRegex($type);
            } else {
                $pattern .= $part;
            }
        }
        $pattern .= "/";
        return $pattern;
    }
    public static function hasMatch($path){
        foreach(self::$routes as $route){
            if(filter::validate($path,$route["pattern"])){
                self::$route = $route["options"];
                self::$match = true;
                return self::$match;
            }
        }
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
