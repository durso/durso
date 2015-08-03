<?php

/**
 * Description of inline
 *
 * @author durso
 */
namespace library\layout\elements;
use library\layout\elements\element;

class inline extends element{

    
    public function __construct($value,$className = array(),$tag = "span") {
        $this->attributes["class"] = $className;
        $this->tag = $tag;
        $this->closeTag = true;
        $this->value = $value;
    }


}
