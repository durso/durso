<?php
/**
 * This class wraps a list of elements
 *
 * @author durso
 */
namespace library\layout\elements;
use library\layout\elements\element;

class group extends element{

    
    public function __construct($className = array("form-group"),$tag = "div") {
        $this->attributes["class"] = $className;
        $this->tag = $tag;
        $this->closeTag = true;
    }


}

