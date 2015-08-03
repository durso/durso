<?php
/**
 * Description of ul
 *
 * @author durso
 */

namespace library\layout\elements;
use library\layout\elements\element;

class ul extends element{

    public function __construct($className = array()) {
        $this->attributes["class"] = $className;
        $this->tag = "ul";
        $this->closeTag = true;
    }


}