<?php

/**
 * Class for html singleton tags
 *
 * @author durso
 */
namespace library\layout\elements;
use library\layout\elements\element;

class singleton extends element{

    
    public function __construct($className = array(),$tag = "br") {
        $this->attributes["class"] = $className;
        $this->tag = $tag;
    }


}