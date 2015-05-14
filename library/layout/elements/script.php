<?php
/**
 * Description of script
 *
 * @author durso
 */
namespace library\layout\elements;
use library\layout\elements\element;


class script {
    private static $controller;
    private static $action;
    private static $script = "";
    private static $response = array();
    
    public static function setController($controller){
        self::$controller = $controller;
    }
    public static function setAction($action){
        self::$action = $action;
    }
  
    public static function event(element $element,$event){
        self::bind($event,$element->getId());

    }
    private static function bind($event,$id){
        if($event == 'click'){
            self::$script = "jQuery('#$id').$event(function(){%s});";
            self::$script = self::ajax($event);
        }
    }
    private static function ajax($event){
        $request = "jQuery.ajax({url:'/".self::$controller."/".self::$action."',data:{event:'$event'},dataType:'json',context:this})";    
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