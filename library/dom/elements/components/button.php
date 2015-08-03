<?php

/**
 * Description of button
 *
 * @author durso
 */
namespace library\dom\elements\components;
use library\dom\elements\components\intext;


class button extends intext{
    
    public function __construct($value = false, $type = "submit",$class = array("btn","btn-default")) {
        parent::__construct("button",$value);
        if($type){
            $this->attributes["type"] = $type;
        }
        $this->attributes["class"] = $class;
    }

    
   
    
    
}
