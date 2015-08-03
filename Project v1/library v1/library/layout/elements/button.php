<?php

/**
 * Description of button
 *
 * @author durso
 */
namespace library\layout\elements;
use library\layout\elements\element;
use app\request;

class button extends element {
    
    public function __construct($value, $type = "submit",$class = array("btn","btn-default"),$tag = "button") {
        $this->value = $value;
        if($type){
            $this->attributes["type"] = $type;
        }
        $this->tag = $tag;
        $this->attributes["class"] = $class;
        $this->setCloseTag();
        $this->setId("button");
    }
    public function setCloseTag(){
        if($this->tag != "input"){
            $this->closeTag = true;
        }
    }
   
    
    
}
