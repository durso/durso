<?php
/**
 * Description of script
 *
 * @author durso
 */
namespace library\dom\elements\components;
use library\dom\elements\paired;
use library\dom\elements\components\text;
use library\dom\object;




class script extends paired{
    
    public function __construct($src = false) {
        parent::__construct();
        $this->tag = "script";
        if($src){
            $this->attributes["src"] = $src;
        }
    }


    public function addComponent(object $component){
        if($component instanceof text){
           parent::addComponent($component); 
        }
    }
  
    /*
    public function event(element $element,$event){
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
   

    
    public static function getScript(){
        $script = self::getSrc();
        $script .= "<script>jQuery(document).ready(function(){";
        foreach(self::$listScript as $jquery){
            $script .= $jquery;
        }
        $script .= " });</script>";
        return $script;
    }
    public static function getResponse(){
        return json_encode(self::$response);
    }
     */
}