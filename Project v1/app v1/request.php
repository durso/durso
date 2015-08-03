<?php

/**
 * Description of request
 *
 * @author durso
 */
namespace app;
use app\router;
use library\filter;
use app\controller\apologize;

class request {
    public static $method;
    public static $params = array();
    public static $uri = array();
    protected static $url;
    protected static $path;
    protected static $baseUrl;  
    protected static $errorMsg = null;
    

    
    public static function init(){
        try{
            self::$method = $_SERVER["REQUEST_METHOD"];
            self::setPath();
            self::setParams();
            self::setUri();
        }catch(\Exception $e){
            $msg = $e->getMessage();
            self::error($msg);
        }    
        
    }
    protected static function setParams(){
        $params = array();
        switch(self::$method) {  
            case 'GET':    $params = $_GET;
                           break;  
            case 'POST':   $params = $_POST;
                           break;
            default:       $params = array();
        }
        if(isset($params["crm"]) && !empty($params["crm"])){
            if(filter::validate($params["crm"],filter::getRegex("int",true))){
                self::$params["crm"] = $params["crm"]; 
            }
        }
        if(isset($params["nome"]) && !empty($params["nome"])){
            if(filter::validate($params["nome"],filter::getRegex("name",true))){
                self::$params["nome"] = $params["nome"];
            }
        }
        if(isset($params["estado"]) && !empty($params["estado"])){
            if(filter::validate($params["estado"],filter::getRegex("int",true))){
                self::$params["estado"] = $params["estado"];
            }
        }
        if(isset($params["id"]) && !empty($params["id"])){
            if(filter::validate($params["id"],filter::getRegex("int",true))){
                self::$params["id"] = $params["id"];
            } 
        }
        if(isset($params["last_id"]) && !empty($params["last_id"])){
            if(filter::validate($params["last_id"],filter::getRegex("int",true))){
                self::$params["last_id"] = $params["last_id"];
            } 
        }
        if(self::isAjax()){
            if(isset(self::$params["uid"])){
                if(!filter::validate(self::$params["uid"],filter::getRegex("id",true))){
                   throw new \Exception("UID: Invalid string format");
                }
            }
            if(isset(self::$params["event"])){
                if(!filter::validate(self::$params["event"],filter::getRegex("alpha",true))){
                    throw new \Exception("Event: Invalid string format");
                }
            }
        }
    }
    
    protected static function setPath(){
        preg_match(filter::getRegex("path",true),$_SERVER["REQUEST_URI"],$matches);
        self::$path = $matches[0];
    }
    
    protected static function setUri(){
        if(self::$path == "/"){
            self::$uri["controller"] = "index";
            self::$uri["action"] = "index";
            return;
        }
        $pathList = explode('/', self::$path); 
        array_shift($pathList);
        $pathList = self::setController($pathList);
        $pathList = self::setAction($pathList);
        $count = count($pathList);
        $i = 0;
        while($i < $count){
            if(array_key_exists($i + 1, $pathList)){
                self::$uri[$pathList[$i]] = $pathList[++$i];
            }
            $i++;
        }
        if(router::hasMatch()){
            self::$uri = array_merge(self::$uri,router::getMatch());
        }
    }
    protected static function setController($pathList){
        if(router::checkMatch(self::$path)){
           self::$uri["controller"] = router::getController();
        } else {
            self::$uri["controller"] = $pathList[0];
            array_shift($pathList);
        }
        return $pathList;
    }
    protected static function setAction($pathList){
        if(router::hasMatch()){
           self::$uri["action"] = router::getAction();
        } else {
            if(count($pathList)){
                self::$uri["action"] = $pathList[0];
                array_shift($pathList);
            } else {
                self::$uri["action"] = "index";
            }
        }
        return $pathList;
    }
    public static function isAjax(){
        return (!empty($_SERVER['HTTP_X_REQUESTED_WITH']) && strtolower($_SERVER['HTTP_X_REQUESTED_WITH']) == 'xmlhttprequest');
    }
    protected function error($msg){
        self::$uri["controller"] = "apologize";
        if(self::isAjax()){
            self::$uri["action"] = "ajax";
        } else {
            self::$uri["action"] = "index";
        }
        self::$errorMsg = $msg;
    }
    public static function hasError(){
        return !is_null(self::$errorMsg);
    }
    public static function errorMsg(){
        return self::$errorMsg;
    }
    public static function isPost(){
        return self::$method == 'POST';
    }
    public static function redirectError($error){
        self::error($error);
        return new apologize(self::$uri["action"], self::$uri,$error);
    }
    
}