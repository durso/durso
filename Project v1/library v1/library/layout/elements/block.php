<?php


/**
 * Description of block
 *
 * @author durso
 */
namespace library\layout\elements;
use library\layout\elements\element;

class block extends element{

    
    public function __construct($value,$className = array(),$tag = "h1") {
        if(!empty($className)){
            $this->attributes["class"] = $className;
        }
        $this->tag = $tag;
        $this->closeTag = true;
        $this->value = $value;
    }


}