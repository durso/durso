<?php
/**
 * Description of script
 *
 * @author durso
 */
namespace library\layout\elements;
use library\layout\elements\element;
use app\request;



class script {
    private static $script = "";
    private static $response = array();
    private static $startScript = false;
    
    public static function start(){
        self::$startScript = true;
    }
    public static function isActive(){
        return self::$startScript;
    }
    public static function setController($controller){
        self::$controller = $controller;
    }
    public static function setAction($action){
        self::$action = $action;
    }
  
    public static function event(element $element,$event){
        self::bind($event,$element->getUid());

    }
    private static function bind($event,$uid){
        if($event == 'click'){
            self::$script = "jQuery('$uid').$event(function(){%s});";
            self::$script = self::ajax($event,$uid);
        }
    }
    private static function ajax($event,$uid){
        $request = "jQuery.ajax({url:'/".request::$uri["controller"]."/".request::$uri["action"]."',data:{event:'$event',uid:'$uid'},dataType:'json',context:this})";    
        $done = ".done(function(result){%s});";
        $callback = "var self = this;"
                . "jQuery.each(result,function(i,item){"
                .   "if(item.context != \"this\"){"
                .       "if('key' in item){"
                .           "eval(sprintf(\"jQuery(%s).%s('%s','%s');\",item.context,item.method,item.key,item.value));"
                .       "}else{"
                .           "eval(sprintf(\"jQuery(%s).%s('%s');\",item.context,item.method,item.value));"
                .       "}"
                .   "} else {"
                .       "if('key' in item){"
                .           "eval(\"jQuery(self)\"+sprintf(\".%s('%s','%s');\",item.method,item.key,item.value));"
                .       "}else{"
                .           "eval(\"jQuery(self)\"+sprintf(\".%s('%s');\",item.method,item.value));"
                .       "}"
                .   "}"
                . "});";
        $done = sprintf($done,$callback);
        $ajax = $request.$done;
        return sprintf(self::$script,$ajax);     
    }

    public static function addValue($value,$context,$method){
        self::$response[] = array("context" => $context,"method" => $method, "value" => sprintf($value));
    }
    public static function addKeyValue($key,$value,$context,$method){
        self::$response[] = array("context" => $context,"method" => $method,"key" =>$key, "value" => $value);
    }
    public static function getScript(){
        return "<script>jQuery(document).ready(function(){".self::$script." });</script>";
    }
    public static function getResponse(){
        return json_encode(self::$response);
    }
}