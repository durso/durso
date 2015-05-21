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
    private static $listScript = array();
    
    public static function start(){
        self::$startScript = true;
    }
    public static function isActive(){
        return self::$startScript;
    }

  
    public static function event(element $element,$event){
        if(!array_key_exists($event, self::$listScript)){
            self::bind($event);
        }
    }

    private static function bind($event){
        if($event == 'click'){
            self::$script = "jQuery('.click').$event(function(){%s});";
            self::$script = self::ajax($event);
            self::$listScript[$event] = self::$script;
        }
    }
    private static function ajax($event){
        $request = "var id = '#'+jQuery(this).attr('id');"
                .   "jQuery.ajax({url:'/".request::$uri["controller"]."/".request::$uri["action"]."',data:{event:'$event',uid:id},dataType:'json',context:this})";    
        $done = ".done(function(result){%s});";
        $callback = "var self = this;"
                . "jQuery.each(result,function(i,item){"
                .       "runResponse(item,i,self);" 
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
        $jquery = "<script>jQuery(document).ready(function(){";
        foreach(self::$listScript as $script){
            $jquery .= $script;
        }
        $jquery .= " });</script>";
        return $jquery;
    }
    public static function getResponse(){
        return json_encode(self::$response);
    }
}